from typing import Callable, List, Optional


def generate_masked_orthogonal_rank_groups(world_size: int, parallel_size: List[int],
                                           mask: List[bool]) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (sum([x * y for x, y in zip(idx, stride[:-1])
                    ]) == index), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride) +
                inner_product(decomposed_group_idx, unmasked_stride))
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0) -> None:
        assert (ep == 1 or cp == 1), "Both EP and CP > 1 in not allow in one rank generator. \
            CP is only included in default RankGenerator, and EP only in expert RankGenerator."

        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                                   f"specified the order ({self.order}).")
            elif name not in order:
                order = order + '-' + name

        self.order = order
        self.ordered_size = []

        for token in order.split('-'):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """Create a mask for the specified tokens based on the given order.

        Args:
            order (str): The order of parallelism types (e.g., 'tp-dp-pp').
            token (str): The specific parallelism types to include in the mask,
                         separated by hyphens (e.g., 'tp-dp').
        """
        ordered_token = order.split('-')
        token_list = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks


def generator_wrapper(rg, eg, group_type, is_expert=False, **kwargs):
    """The `RankGenerator` class produces a hyper-rectangle for a given set of
    tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
    in addition to the default decoder, we essentially instantiate two `RankGenerator`
    classes to construct the parallelism for each module separately, and we then have
    to stitch them together for the right groups. For now, this means pp and tp-pp."""
    if is_expert:
        d_ranks = eg.get_ranks(group_type, **kwargs)
    else:
        d_ranks = rg.get_ranks(group_type, **kwargs)

    for x in d_ranks:
        yield x
    return


def main():
    tp = 2
    pp = 4
    cp = 2
    model_size = tp * pp * cp
    world_size = 32
    assert world_size % (tp * pp * cp) == 0, f'{world_size=} %!=0 {tp=} {pp=} {cp=}'
    dp = world_size // (tp * pp * cp)
    print(f'{dp=}, {tp=}, {pp=}, {cp=}, {world_size=}')

    order = 'tp-cp-ep-dp-pp'

    rg = RankGenerator(
        tp=tp,
        ep=1,
        dp=dp,
        pp=pp,
        cp=cp,
        order=order,
        rank_offset=0,
    )
    #print(rg.get_ranks('pp'))

    # moe
    ep = 2  # ep==cp
    eg = RankGenerator(
        tp=tp,
        ep=ep,
        dp=dp,
        pp=pp,
        cp=1,
        order=order,
        rank_offset=0,
    )
    #print(eg.get_ranks('pp'))
    assert eg.get_ranks('pp') == rg.get_ranks('pp')

    print('*' * 100)
    print('init global vars:')

    print('dp: ')
    for ranks in generator_wrapper(rg, eg, 'dp'):
        print(ranks)
    print(rg.get_ranks('dp'))

    print('dp-cp:')
    print(rg.get_ranks('dp-cp'))

    print('cp:')
    print(rg.get_ranks('cp'))

    print('tp:')
    print(rg.get_ranks('tp'))

    print('pp:')
    print(rg.get_ranks('pp'))

    print('*' * 100)
    print('Expert dp: ')
    print(eg.get_ranks('dp'))

    print('Expert ep: ')
    print(eg.get_ranks('ep'))

    print('Expert tp: ')
    print(eg.get_ranks('tp'))


if __name__ == '__main__':
    main()
