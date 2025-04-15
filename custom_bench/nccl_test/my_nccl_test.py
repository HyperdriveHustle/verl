import os
import torch
import torch.distributed as dist
import time
import numpy as np
import argparse
from datetime import timedelta

# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \  # for master IP, node rank must be 0
#     --master_addr=10.0.0.1 \
#     --master_port=29500 \
#     nccl.py \
#     --op send-recv

# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 nccl.py --op send-recv


def format_size(size_in_bytes):
    """Convert bytes into a human-readable format (KB, MB, GB, etc.)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = size_in_bytes
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def get_sizes(args):
    sizes = [
        #1024,  # 1 KB
        1024 * 1024,  # 1 MB
        8 * 1024 * 1024,  # 8 MB
        64 * 1024 * 1024,  # 64 MB
        128 * 1024 * 1024,  # 128 MB
        512 * 1024 * 1024,  # 512 MB
        1024 * 1024 * 1024,  # 1 GB
        # Add larger sizes cautiously, ensure sufficient GPU memory
        # 2 * 1024 * 1024 * 1024, # 2 GB
        # 4 * 1024 * 1024 * 1024, # 4 GB
    ]
    return sizes


def setup(backend='nccl'):
    """Initialize distributed environment based on environment variables."""
    timeout = timedelta(seconds=60)
    dist.init_process_group(
        backend=backend,
        timeout=timeout,
        #init_method='env://',  # 使用环境变量初始化
    )


def measure_all_reduce_bandwidth(args):
    """
    Measure bandwidth using all-reduce operation
    """
    setup()

    # Get local rank and device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
    # device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)

    # Sizes to test (in bytes)
    sizes = get_sizes(args)

    # Warmup iterations
    warmup_iters = 20
    test_iters = args.iters

    #print(f"Rank {rank}: Starting All-Reduce Bandwidth Test")
    dist_rank = dist.get_rank()
    dist_local_rank = rank % torch.cuda.device_count()
    print(
        f'all-reduce {rank=} {local_rank=} {world_size=} {dist_rank=} {dist_local_rank=} {visible_gpu=}'
    )

    for size in sizes:
        # Create tensor on GPU
        # tensor = torch.ones(size // 4, dtype=torch.float32, device=device)
        num_elements = size // (torch.finfo(torch.float32).bits // 8)
        #tensor = torch.rand(num_elements, device=f'cuda:{local_rank}')
        tensor = torch.rand(num_elements, device=f'cuda:{local_rank}')

        # Warmup iterations
        for _ in range(warmup_iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()

        # Actual test iterations
        start_time = time.time()
        for _ in range(test_iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        end_time = time.time()

        torch.cuda.empty_cache()

        # Calculate bandwidth
        total_data_transferred = size * dist.get_world_size() * test_iters
        duration = end_time - start_time
        bandwidth = (total_data_transferred /
                     (1024 * 1024 * 1024)) / duration  # GB/s

        if local_rank == args.r:
            print(
                f"{rank=}, {local_rank=}, All-Reduce Bandwidth for {format_size(size)}"
            )
            print(f"  Total time: {duration:.4f} seconds")
            print(f"  Bandwidth: {bandwidth:.2f} GB/s")
    dist.destroy_process_group()


def measure_send_recv_bandwidth(args):
    """
    Measure bandwidth using send and recv operations
    """
    # Initialize the distributed environment
    setup()

    # Get local rank and device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)

    # Sizes to test (in bytes)
    sizes = get_sizes(args)

    # Warmup iterations
    warmup_iters = 20
    test_iters = args.iters

    if rank == 0:
        p1 = world_size - 1
    else:
        p1 = rank - 1

    if rank == world_size - 1:
        p2 = 0
    else:
        p2 = rank + 1

    dist_rank = dist.get_rank()
    dist_local_rank = rank % torch.cuda.device_count()
    print(
        f'send-recv {rank=} {local_rank=} {world_size=} {visible_gpu=} {dist_rank=} {dist_local_rank=} {p1=} {p2=}'
    )

    for size in sizes:
        # Create tensors on GPU
        send_tensor = torch.rand(size // 4, dtype=torch.float32, device=device)
        recv_tensor = torch.zeros(size // 4,
                                  dtype=torch.float32,
                                  device=device)

        # Warmup iterations
        for _ in range(warmup_iters):
            # NOTE separate odd/even to avoid deadlock
            if rank % 2 == 0:
                dist.recv(recv_tensor, src=p1)
            else:
                dist.send(send_tensor, dst=p2)

            if rank % 2 == 0:
                dist.send(send_tensor, dst=p1)
            else:
                dist.recv(recv_tensor, src=p2)
        torch.cuda.synchronize()
        dist.barrier()

        # Actual test iterations
        start_time = time.time()
        for _ in range(test_iters):
            # NOTE separate odd/even to avoid deadlock
            if rank % 2 == 0:
                dist.recv(recv_tensor, src=p1)
            else:
                dist.send(send_tensor, dst=p2)

            if rank % 2 == 0:
                dist.send(send_tensor, dst=p1)
            else:
                dist.recv(recv_tensor, src=p2)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate bandwidth
        total_data_transferred = size * test_iters
        duration = end_time - start_time
        bandwidth = (total_data_transferred /
                     (1024 * 1024 * 1024)) / duration  # GB/s

        # sr may have individual variances?
        if local_rank == args.r:
            print(
                f"{rank=}, {local_rank=}, send-recv Bandwidth for {format_size(size)}"
            )
            print(f"  Total time: {duration:.4f} seconds")
            print(f"  Bandwidth: {bandwidth:.2f} GB/s")

    dist.destroy_process_group()


def measure_send_recv_bandwidth2(args):
    '''
    measure cross machine bw,
        e.g. ranks 0-7 on machine 0
        e.g. ranks 8-15 on machine 1
    '''
    assert args.stride is not None
    setup()

    # Get local rank and device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f'cuda:{local_rank}')
    stride = args.stride

    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
    torch.cuda.set_device(local_rank)

    # Sizes to test (in bytes)
    sizes = get_sizes(args)

    # Warmup iterations
    warmup_iters = 20
    test_iters = args.iters

    peer = (rank + stride) % world_size

    dist_rank = dist.get_rank()
    dist_local_rank = rank % torch.cuda.device_count()
    print(
        f'send-recv2 {rank=} {local_rank=} {world_size=} {visible_gpu=} {dist_rank=} {dist_local_rank=} {peer=}'
    )

    for size in sizes:
        # Create tensors on GPU
        send_tensor = torch.rand(size // 4, dtype=torch.float32, device=device)
        recv_tensor = torch.zeros(size // 4,
                                  dtype=torch.float32,
                                  device=device)

        # Warmup iterations
        for _ in range(warmup_iters):
            # NOTE separate odd/even to avoid deadlock
            if rank < 8:
                dist.send(send_tensor, dst=peer)
            else:
                dist.recv(recv_tensor, src=peer)

            if rank < 8:
                dist.recv(recv_tensor, src=peer)
            else:
                dist.send(send_tensor, dst=peer)
        torch.cuda.synchronize()
        dist.barrier()

        # Actual test iterations
        start_time = time.time()
        for _ in range(test_iters):
            # NOTE separate odd/even to avoid deadlock
            if rank < 8:
                dist.send(send_tensor, dst=peer)
            else:
                dist.recv(recv_tensor, src=peer)

            if rank < 8:
                dist.recv(recv_tensor, src=peer)
            else:
                dist.send(send_tensor, dst=peer)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate bandwidth
        total_data_transferred = size * test_iters
        duration = end_time - start_time
        bandwidth = (total_data_transferred /
                     (1024 * 1024 * 1024)) / duration  # GB/s

        # sr may have individual variances?
        #if rank % 2 == 0:
        if local_rank == args.r:
            print(
                f"{rank=}, {local_rank=}, send-recv Bandwidth for {format_size(size)}"
            )
            print(f"  Total time: {duration:.4f} seconds")
            print(f"  Bandwidth: {bandwidth:.2f} GB/s")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="NCCL Bandwidth Measurement")
    parser.add_argument('--op',
                        choices=['all-reduce', 'send-recv', 'send-recv2'],
                        default='all-reduce',
                        help='Type of bandwidth test')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stride', type=int, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.op == 'all-reduce':
        measure_all_reduce_bandwidth(args)
    elif args.op == 'send-recv':
        measure_send_recv_bandwidth(args)
    elif args.op == 'send-recv2':
        measure_send_recv_bandwidth2(args)
    else:
        raise RuntimeError(f'unknown op {args.op}')


if __name__ == "__main__":
    main()
