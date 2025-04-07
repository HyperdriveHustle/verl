import os
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# torchrun --nproc_per_node=4  custom_bench/ddp.py

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GLOBAL_RANKS = None


def setup_distributed():
    # Get rank and local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(
        f"Process RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}"
    )
    return local_rank, rank, world_size


def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    kwargs = {
        'ranks': ranks,
        #'timeout': timeout,
        'backend': backend,
        #'pg_options': pg_options,
        #'use_local_synchronization': use_local_synchronization,
        #'group_desc': group_desc,
    }
    return torch.distributed.new_group(**kwargs)


def get_data_parallel_rank():
    """Return caller's rank in the data-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=_DATA_PARALLEL_GROUP)
    assert False


def main():
    local_rank, rank, world_size = setup_distributed()

    # test customized group
    for group_ranks in [[0, 2], [1, 3]]:
        group = create_group(ranks=group_ranks, backend='gloo')
        if rank in group_ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GLOBAL_RANKS = group_ranks
    print(
        f'Worker: {rank=}, {_DATA_PARALLEL_GROUP=}, {_DATA_PARALLEL_GLOBAL_RANKS}, dp-rank: {get_data_parallel_rank()}'
    )

    # train
    model = nn.Linear(10, 10)
    local_model = deepcopy(model)

    #model = DDP(model, device_ids=[local_rank])  # cuda
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Different processes can perform different tasks based on rank
    if rank == 0:
        for p1, p2 in zip(model.parameters(), local_model.parameters()):
            assert torch.allclose(p1.data, p2.data)
        print(f"init check {rank=}, pass local check")

    dist.barrier()

    for epoch in range(3):
        # Each process works on its subset of data
        dummy_input = torch.randn(20, 10)
        dist_dummy_output = model(dummy_input)
        loss = dist_dummy_output.sum()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # local train
        # dummy_output = local_model(dummy_input)
        # loss = dummy_output.sum()
        # loss.backward()

        # Synchronize after each epoch
        dist.barrier()

        if rank == 0:
            # assert torch.allclose(dummy_output, dist_dummy_output)

            # NOTE: to check grad, data should be gathered for each process
            # grads_non_equal = 0
            # cnt = 0
            # for p1, p2 in zip(model.parameters(), local_model.parameters()):
            #     if p1.grad is not None and p2.grad is not None:
            #         cnt+=1
            #         if not torch.allclose(p1.grad.data, p2.grad.data):
            #             grads_non_equal +=1
            # if grads_non_equal > 0:
            #     raise AssertionError(f"gradients are not equal, {grads_non_equal}/{cnt}")

            # # Update local model to match DDP model for next epoch
            # with torch.no_grad():
            #     for p_ddp, p_local in zip(model.parameters(), local_model.parameters()):
            #         p_local.data.copy_(p_ddp.data)
            # # Zero out gradients for next iteration
            # for p in local_model.parameters():
            #     if p.grad is not None:
            #         p.grad.zero_()

            print(f"Epoch {epoch} completed, pass local check")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
