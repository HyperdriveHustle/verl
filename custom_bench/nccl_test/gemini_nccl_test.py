import os
import torch
import torch.distributed as dist
import time
import numpy as np
import argparse
from datetime import timedelta

# Example torchrun commands:
# Single Op (e.g., all-gather on 2 nodes, 8 GPUs each):
# Node 0:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all-gather --iters 100 -r 0
# Node 1:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all-gather --iters 100 -r 0

# Run all Ops sequentially:
# Node 0:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all --iters 100 -r 0
# Node 1:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all --iters 100 -r 0

# Constants
FLOAT32_BYTES = torch.finfo(torch.float32).bits // 8
DEFAULT_TIMEOUT_SECONDS = 60  # Increased timeout for potentially slower operations or larger clusters


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Communication Bandwidth Measurement")
    parser.add_argument(
        '--op',
        type=str,
        choices=[
            #'send-recv',      # Point-to-point ring
            'all-reduce',
            'all-gather',
            'reduce-scatter',
            'broadcast',
            'all'  # Run all supported tests sequentially
        ],
        required=True,
        help='Type of communication operation(s) to benchmark.')
    parser.add_argument('--iters',
                        type=int,
                        default=100,
                        help='Number of measurement iterations.')
    parser.add_argument('--warmup',
                        type=int,
                        default=20,
                        help='Number of warmup iterations.')
    parser.add_argument('-r',
                        type=int,
                        default=0,
                        help='Local rank responsible for printing results.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for tensor initialization.')
    # Potentially add --sizes argument later if needed:
    # parser.add_argument('--sizes', type=str, default=None, help='Comma-separated list of sizes (e.g., "1KB,1MB,1GB")')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend to use.')

    return parser.parse_args()


def format_size(size_in_bytes):
    """Convert bytes into a human-readable format (KB, MB, GB, etc.)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_in_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def get_sizes(args):
    """Defines the tensor sizes (in bytes) to test."""
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
    # Allow overriding sizes via command line if needed in the future
    # if args.sizes:
    #     sizes = [parse_size_str(s) for s in args.sizes.split(',')]
    return sizes


def setup(backend='nccl'):
    if dist.is_initialized():
        print(
            "Warning: Distributed process group already initialized. Skipping setup."
        )
        return

    # Ensure necessary env vars are set
    required_env_vars = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"
    ]
    for var in required_env_vars:
        if var not in os.environ:
            raise RuntimeError(
                f"Required environment variable {var} is not set. "
                "Please launch using torchrun or similar.")

    rank = os.environ.get("RANK", "N/A")
    local_rank = os.environ.get("LOCAL_RANK", "N/A")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")

    timeout = timedelta(seconds=DEFAULT_TIMEOUT_SECONDS)
    print(
        f"{rank=} {local_rank=} {visible_gpu=} {world_size=} Initializing process group with backend '{backend}' and timeout {timeout}..."
    )
    try:
        dist.init_process_group(
            backend=backend,
            timeout=timeout,
            # init_method='env://' is default and recommended
        )
        print(
            f"{rank=} {local_rank=} {visible_gpu=} {world_size=} Process group initialized successfully."
        )
        # Optional: Add a barrier here to ensure all processes initialized before proceeding
        dist.barrier()
        print(
            f"{rank=} {local_rank=} {visible_gpu=} {world_size=} Initial barrier passed."
        )
    except Exception as e:
        print(
            f"[Rank {rank}, LocalRank {local_rank}] {visible_gpu=} Error during init_process_group: {e}"
        )
        # Consider cleanup or re-raising depending on desired fault tolerance
        raise


def cleanup():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        print("Cleaning up distributed process group...")
        dist.destroy_process_group()
        print("Process group destroyed.")
    else:
        print("Cleanup: Distributed process group not initialized.")


def print_test_header(op_name, rank, local_rank, world_size, visible_gpu):
    """Prints a standardized header for each test."""
    dist_rank = dist.get_rank() if dist.is_initialized() else rank
    dist_local_rank = local_rank  # Assuming local_rank derived correctly
    header = (
        f"--- Starting {op_name} Bandwidth Test --- \n"
        f"Global Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}\n"
        f"PyTorch Dist Rank: {dist_rank}, Visible GPUs: {visible_gpu}\n"
        f"PyTorch Dist Local Rank (derived): {dist_local_rank}")
    # Only print from the designated reporting rank (or rank 0 if not specified)
    reporting_rank = int(os.environ.get("REPORTING_RANK",
                                        0))  # Use env var if needed
    if rank == reporting_rank:
        print(header)
    dist.barrier()  # Ensure header is printed before tests start across ranks


# --- Measurement Functions ---


def measure_op_bandwidth(
    args,
    op_name,
    comm_op_func,
    bandwidth_factor=1.0,
    setup_tensors_func=None,
    requires_input_list=False,
    requires_output_list=False,
):
    """
    Generic function to measure bandwidth for a given communication operation.

    Args:
        args: Command line arguments.
        op_name (str): Name of the operation (e.g., 'all-reduce').
        comm_op_func (callable): The torch.distributed communication function (e.g., dist.all_reduce).
        bandwidth_factor (float): Multiplier for bandwidth calculation (e.g., 2.0 for all-reduce).
        setup_tensors_func (callable, optional): Custom function to create tensors if needed.
        requires_input_list (bool): If the comm_op requires an input list (e.g., reduce_scatter).
        requires_output_list (bool): If the comm_op requires an output list (e.g., all_gather).
    """
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        print_test_header(op_name, rank, local_rank, world_size, visible_gpu)

        sizes = get_sizes(args)
        warmup_iters = args.warmup
        test_iters = args.iters

        results = {}  # Store results {size: bandwidth}

        for size in sizes:
            num_elements = size // FLOAT32_BYTES
            if num_elements == 0:
                if local_rank == args.r:
                    print(
                        f"{rank=} {local_rank=} Skipping size {format_size(size)} as it results in 0 elements."
                    )
                continue

            # --- Tensor Creation ---
            tensor = None
            input_list = None
            output_list = None
            output_tensor = None

            try:
                if setup_tensors_func:
                    # Custom tensor setup if provided
                    tensor, input_list, output_list, output_tensor = setup_tensors_func(
                        size, num_elements, device, world_size)
                else:
                    # Default: create a single tensor
                    tensor = torch.rand(num_elements,
                                        dtype=torch.float32,
                                        device=device)
                    if requires_input_list:
                        # Needed for reduce_scatter
                        input_list = [
                            torch.rand(num_elements,
                                       dtype=torch.float32,
                                       device=device)
                            for _ in range(world_size)
                        ]
                        # Output tensor for reduce_scatter
                        output_tensor = torch.empty(num_elements,
                                                    dtype=torch.float32,
                                                    device=device)
                    if requires_output_list:
                        # Needed for all_gather
                        output_list = [
                            torch.empty(num_elements,
                                        dtype=torch.float32,
                                        device=device)
                            for _ in range(world_size)
                        ]
                        output_tensor = output_list  # For clarity in comm_op call
                    if not requires_input_list and not requires_output_list:
                        output_tensor = tensor  # For ops like broadcast/all_reduce

            except torch.cuda.OutOfMemoryError:
                if local_rank == args.r:
                    print(
                        f"OOM ERROR: Cannot allocate tensor(s) for size {format_size(size)} on rank {rank} (GPU {local_rank}). Skipping this size."
                    )
                dist.barrier(
                )  # Ensure all ranks acknowledge OOM before continuing/exiting
                # Clean up any partially allocated tensors if possible
                del tensor, input_list, output_list, output_tensor
                torch.cuda.empty_cache()
                continue  # Skip to next size
            except Exception as e:
                if local_rank == args.r:
                    print(
                        f"ERROR during tensor creation for size {format_size(size)} on rank {rank}: {e}"
                    )
                dist.barrier()
                raise  # Propagate other unexpected errors

            # --- Warmup ---
            dist.barrier()  # Ensure tensors created everywhere before warmup
            if local_rank == args.r:
                print()
                print(f"{rank=} Warming up for size {format_size(size)}...")
            for i in range(warmup_iters):
                try:
                    if requires_input_list:
                        comm_op_func(output_tensor, input_list)
                    elif requires_output_list:
                        comm_op_func(output_list, tensor)
                    elif op_name == 'broadcast':  # Special case for broadcast signature
                        comm_op_func(tensor, src=0)
                    else:  # Default case (e.g., all_reduce)
                        comm_op_func(tensor)
                except Exception as e:
                    if rank == args.r: print(f"Warmup Iter {i} Error: {e}")
                    dist.barrier()
                    raise
            torch.cuda.synchronize()
            dist.barrier()
            if local_rank == args.r:
                print(f"{rank=} Warmup complete.")

            # --- Measurement ---
            if local_rank == args.r:
                print(
                    f"Starting measurement ({test_iters} iters) for size {format_size(size)}..."
                )
            start_time = time.time()
            for _ in range(test_iters):
                if requires_input_list:
                    comm_op_func(output_tensor, input_list)
                elif requires_output_list:
                    comm_op_func(output_list, tensor)
                elif op_name == 'broadcast':
                    comm_op_func(tensor, src=0)
                else:
                    comm_op_func(tensor)
            torch.cuda.synchronize()  # Ensure GPU work is done
            end_time = time.time()
            dist.barrier()  # Ensure all ranks finished before calculating time

            duration = end_time - start_time

            # --- Bandwidth Calculation ---
            # Base data transferred per iteration (size of the tensor involved per rank)
            # For collectives like all-reduce, reduce-scatter, we use a factor of 2
            # to approximate the data movement (send + receive) in algorithms like ring.
            # For point-to-point, broadcast, all-gather, it's typically the size itself.
            total_data_transferred = size * bandwidth_factor * test_iters
            bandwidth_gbps = (
                total_data_transferred /
                (1024 * 1024 * 1024)) / duration if duration > 0 else 0

            if local_rank == args.r:
                print(
                    f"  Size: {format_size(size)}\n"
                    f"    Total time: {duration:.4f} seconds\n"
                    f"    Bandwidth ({'Algo' if bandwidth_factor > 1 else 'Per Rank'}): {bandwidth_gbps:.2f} GB/s"
                )
                results[size] = bandwidth_gbps

            # --- Cleanup per size ---
            del tensor, input_list, output_list, output_tensor
            torch.cuda.empty_cache()
            dist.barrier()

        if local_rank == args.r:
            print(f"--- {op_name} Test Complete ---")
            print()
        return results  # Return results for potential aggregation

    except Exception as e:
        # Ensure cleanup happens even if errors occur mid-test
        local_rank = os.environ.get("LOCAL_RANK", "N/A")
        rank = os.environ.get("RANK", "N/A")
        print(
            f"[Rank {rank}, LocalRank {local_rank}] EXCEPTION in measure_op_bandwidth ({op_name}): {e}"
        )
        import traceback
        traceback.print_exc()


def measure_send_recv_bandwidth(args):
    """Measure bandwidth using send/recv in a ring pattern."""
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        print_test_header("Send-Recv (Ring)", rank, local_rank, world_size,
                          visible_gpu)

        sizes = get_sizes(args)
        warmup_iters = args.warmup
        test_iters = args.iters

        # Determine communication peers for the ring
        p_send = (rank + 1) % world_size  # Send to next rank
        p_recv = (rank - 1 +
                  world_size) % world_size  # Receive from previous rank

        if local_rank == args.r:
            print(
                f"Rank {rank} setup: Sending to {p_send}, Receiving from {p_recv}"
            )
        dist.barrier()

        results = {}

        for size in sizes:
            num_elements = size // FLOAT32_BYTES
            if num_elements == 0:
                if rank == args.r:
                    print(f"Skipping size {format_size(size)}...")
                continue

            try:
                send_tensor = torch.rand(num_elements,
                                         dtype=torch.float32,
                                         device=device)
                recv_tensor = torch.zeros(num_elements,
                                          dtype=torch.float32,
                                          device=device)
            except torch.cuda.OutOfMemoryError:
                if local_rank == args.r:
                    print(
                        f"OOM ERROR: Cannot allocate tensors for size {format_size(size)} on rank {rank}. Skipping."
                    )
                dist.barrier()
                del send_tensor, recv_tensor
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                if local_rank == args.r:
                    print(
                        f"ERROR during tensor creation for size {format_size(size)}: {e}"
                    )
                dist.barrier()
                raise

            # --- Warmup ---
            dist.barrier()
            if local_rank == args.r:
                print()
                print(f"{rank=} Warming up for size {format_size(size)}...")
            reqs = []
            for _ in range(warmup_iters):
                # Simple non-blocking approach, could also use blocking with odd/even separation
                # Non-blocking might show slightly different performance characteristics
                req_send = dist.isend(send_tensor, dst=p_send)
                req_recv = dist.irecv(recv_tensor, src=p_recv)
                reqs.extend([req_send, req_recv])

                # Wait for this pair to complete before next iteration
                for req in [req_send, req_recv]:
                    req.wait()
                # Barrier between iterations might be needed if synchronization issues arise
                # dist.barrier()

            # Ensure all prior ops completed if using non-blocking
            # for req in reqs:
            #     req.wait()
            torch.cuda.synchronize()
            dist.barrier()
            if local_rank == args.r: print("Warmup complete.")

            # --- Measurement ---
            if local_rank == args.r:
                print(
                    f"Starting measurement ({test_iters} iters) for size {format_size(size)}..."
                )
            start_time = time.time()
            # Using blocking send/recv with odd/even separation for deadlock avoidance
            # This is often more stable for benchmarking point-to-point.
            for i in range(test_iters):
                if rank % 2 == 0:
                    dist.send(send_tensor, dst=p_send)
                    dist.recv(recv_tensor, src=p_recv)
                else:
                    dist.recv(recv_tensor, src=p_recv)
                    dist.send(send_tensor, dst=p_send)

                # Optional barrier per iteration for tight sync (can add overhead)
                # dist.barrier()

            torch.cuda.synchronize()
            end_time = time.time()
            dist.barrier()  # Ensure all ranks finished timing section

            duration = end_time - start_time

            # Bandwidth Calculation: Data sent OR received by this process
            total_data_transferred = size * test_iters
            bandwidth_gbps = (
                total_data_transferred /
                (1024 * 1024 * 1024)) / duration if duration > 0 else 0

            if local_rank == args.r:
                print(f"  Size: {format_size(size)}\n"
                      f"    Total time: {duration:.4f} seconds\n"
                      f"    Bandwidth (Per Rank): {bandwidth_gbps:.2f} GB/s")
                results[size] = bandwidth_gbps

            del send_tensor, recv_tensor
            torch.cuda.empty_cache()
            dist.barrier()

        if local_rank == args.r:
            print("--- Send-Recv (Ring) Test Complete ---")
            print()
        return results

    except Exception as e:
        local_rank = os.environ.get("LOCAL_RANK", "N/A")
        rank = os.environ.get("RANK", "N/A")
        print(
            f"[Rank {rank}, LocalRank {local_rank}] EXCEPTION in measure_send_recv_bandwidth: {e}"
        )
        import traceback
        traceback.print_exc()


# --- Main Execution Logic ---


def main():
    args = parse_args()

    # Set seed for reproducibility (affecting torch.rand)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # If numpy is used elsewhere
    # Seed CUDA devices if desired (may have minor performance impact)
    # torch.cuda.manual_seed_all(args.seed)

    setup()

    all_results = {}  # Dictionary to store results if running 'all'

    ops_to_run = []
    if args.op == 'all':
        # Define the order for the 'all' run
        ops_to_run = [
            #'send-recv',
            'broadcast',
            'all-reduce',
            'reduce-scatter',
            'all-gather',
        ]
    else:
        ops_to_run = [args.op]

    for op_choice in ops_to_run:

        results = {}
        if op_choice == 'send-recv':
            #results = measure_send_recv_bandwidth(args)
            raise RuntimeError(f'not impl')
        elif op_choice == 'all-reduce':
            # Bandwidth factor 2.0 for algorithmic bandwidth (Ring Algo approximation)
            results = measure_op_bandwidth(args,
                                           op_name='all-reduce',
                                           comm_op_func=dist.all_reduce,
                                           bandwidth_factor=2.0)
        elif op_choice == 'broadcast':
            # Bandwidth factor 1.0 (data received per non-root node)
            results = measure_op_bandwidth(args,
                                           op_name='broadcast',
                                           comm_op_func=dist.broadcast,
                                           bandwidth_factor=1.0)
        elif op_choice == 'all-gather':
            # Needs output list, BW factor 1.0 (data sent per node)
            results = measure_op_bandwidth(args,
                                           op_name='all-gather',
                                           comm_op_func=dist.all_gather,
                                           bandwidth_factor=1.0,
                                           requires_output_list=True)
        elif op_choice == 'reduce-scatter':
            # Needs input list, BW factor 2.0 (Ring Algo approximation)
            results = measure_op_bandwidth(
                args,
                op_name='reduce-scatter',
                comm_op_func=dist.reduce_scatter,
                bandwidth_factor=2.0,
                requires_input_list=True,
            )
        else:
            raise ValueError(f"Unknown operation: {op_choice}")

        if results:  # Store results if any were generated
            all_results[op_choice] = results

        # Add a small delay and barrier between different ops in 'all' mode
        time.sleep(1)  # Small pause
        dist.barrier()

    # Optional: Print summary if 'all' was run
    if args.op == 'all':
        rank = int(os.environ.get("RANK", 0))
        if rank == args.r:
            print("\n\n===== All Operations Summary =====")
            for op_name, results in all_results.items():
                print(f"\n--- {op_name} Results ---")
                if not results:
                    print("  No results recorded (possibly skipped or OOM).")
                    continue
                for size, bw in results.items():
                    print(
                        f"  Size: {format_size(size)} -> Bandwidth: {bw:.2f} GB/s"
                    )
            print("================================\n")

    cleanup()
    print(f"Rank {os.environ.get('RANK', 'N/A')} finished.")


if __name__ == "__main__":
    main()
