import argparse
import subprocess
import time
from datetime import timedelta
import torch
import torch.distributed as dist

# 假设日志记录器已定义
display_log = print
persistene_log = print


def parse_size(size_str):
    """Parse memory size string like '128M' or '1G' into bytes."""
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
    if size_str[-1].upper() not in units:
        raise ValueError(
            f"Invalid size unit in '{size_str}'. Expected one of {list(units.keys())}."
        )
    unit = size_str[-1].upper()
    number = float(size_str[:-1])
    return int(number * units[unit])


def setup(backend='nccl'):
    """Initialize distributed environment based on environment variables."""
    timeout = timedelta(seconds=300)
    dist.init_process_group(
        backend=backend,
        timeout=timeout,
        init_method='env://',  # 使用环境变量初始化
    )
    rank = dist.get_rank()
    if backend == 'nccl':
        torch.cuda.set_device(rank % torch.cuda.device_count())


def benchmark_nccl_communication(begin_size,
                                 end_size,
                                 factor,
                                 local_rank,
                                 num_tests=10):
    """Conduct NCCL communication performance tests."""
    world_size = dist.get_world_size()
    size = parse_size(begin_size)
    end_size = parse_size(end_size)

    # Warm-up phase: Perform a few rounds of all_reduce to ensure readiness
    tensor = torch.rand(size // (torch.finfo(torch.float32).bits // 8),
                        device=f'cuda:{local_rank}')
    for _ in range(5):
        dist.all_reduce(tensor)
    dist.barrier()  # synchronize all processes

    i = 0
    while size <= end_size:
        i += 1
        num_elements = size // (torch.finfo(torch.float32).bits // 8)
        tensor = torch.rand(num_elements, device=f'cuda:{local_rank}')

        dist.barrier()
        start_time = time.time()

        # Perform all_reduce `num_tests` times and calculate the average duration
        for _ in range(num_tests):
            dist.all_reduce(tensor)
        dist.barrier()
        duration = (time.time() - start_time) / num_tests

        torch.cuda.empty_cache()

        algbw = (size / duration) / 1e9  # in GB/s
        busbw = algbw * (2 * (world_size - 1) / world_size)

        if local_rank == 0:
            display_log(
                f"NCCL Performance Test Round {i}: PASS - Bus Bandwidth: {busbw:.2f} GB/s"
            )
            persistene_log(
                f"NCCL Performance Test Round {i}: PASS - Bus Bandwidth: {busbw:.2f} GB/s"
            )

        size *= factor


def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description=
        "Simulate NCCL test for distributed communication performance testing."
    )
    parser.add_argument("-b",
                        "--begin-size",
                        default="8",
                        help="Starting data size (e.g., 8, 128M, 1G).")
    parser.add_argument("-e",
                        "--end-size",
                        default="128M",
                        help="Ending data size.")
    parser.add_argument("-f",
                        "--factor",
                        type=int,
                        default=2,
                        help="Data size growth factor.")
    args = parser.parse_args()

    try:
        setup(backend='nccl')
        rank = dist.get_rank()
        local_rank = rank % torch.cuda.device_count()
        print(local_rank)
        benchmark_nccl_communication(args.begin_size, args.end_size,
                                     args.factor, local_rank)

    except BaseException as e:
        e_str = str(e).replace('\n', '\\n ').replace('\'', '\"')
        e_str = e_str[:512]
        display_log("Communication TEST: FAIL - NCCL TEST Failed.")
        persistene_log(f"Communication TEST: FAIL - Error: {e_str}")

    cleanup()


if __name__ == "__main__":
    main()
