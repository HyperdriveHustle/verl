import os
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
# torchrun --nproc_per_node=4 custom_bench/ddp.py


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize the process group
    dist.init_process_group(backend="gloo")

    # Set the device
    return local_rank, rank, world_size


def main():
    local_rank, rank, world_size = setup_distributed()
    print(f'worker: {rank=}, {local_rank=}, {world_size=}')
    dist.barrier()

    x = torch.arange(rank * 10, (rank + 1) * 10, dtype=torch.float32)
    if rank == 0:
        print('init')
        print(x)

    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    if rank == 0:
        print('all reduce')
        print(x)
    dist.barrier()

    x = torch.arange(rank * 10, (rank + 1) * 10, dtype=torch.float32)
    out_buf = [torch.zeros((10, )) for _ in range(4)]
    dist.all_gather(out_buf, x)
    if rank == 0 or rank == 1:
        print('all gather')
        print(out_buf)
    dist.barrier()

    # XXX gloo cannot reduce-scatter
    # x = [torch.arange(idx*10, (idx+1)*10, dtype=torch.float32) for idx in range(4)]
    # out_buf = torch.zeros((10,))
    # dist.reduce_scatter(out_buf, x)
    # if rank == 0:
    #     print('reduce scatter')
    #     print(x)
    # dist.barrier()

    x = torch.arange(rank * 4, (rank + 1) * 4, dtype=torch.float32)
    out_buf = torch.zeros(
        1, dtype=torch.float32)  # each rank has 4 elem, reduce to 1
    dist.reduce_scatter_tensor(out_buf, x)
    if rank == 0:
        print('rs-tensor')
        print(out_buf)
    dist.barrier()

    # XXX gloo does not support!
    # [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
    # [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
    # [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
    # [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
    # x = torch.arange(4) + rank * 4
    # x = list(x.chunk(4))
    # out_buf = list(torch.empty([4], dtype=torch.int64).chunk(4))
    # dist.all_to_all(out_buf, x)
    # if rank == 0 :
    #     print('all to all')
    #     print(out_buf)

    x = torch.arange(4) + rank * 4
    out_buf = torch.zeros(4, dtype=torch.int64)
    dist.all_to_all_single(out_buf, x)
    if rank == 0:
        print('all to all single')
        print(x)
        print(out_buf)

    # 2D tensor all-to-all single make sense?
    x = torch.arange(0, 12).reshape(4, 3)
    out_buf = torch.zeros_like(x)
    dist.all_to_all_single(out_buf, x)
    if rank == 0:
        print('all to all single2')
        print(x)
        print(out_buf)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
