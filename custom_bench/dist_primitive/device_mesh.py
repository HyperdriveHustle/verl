import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel, RowwiseParallel

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=0)
    return parser.parse_args()


# Example 1: 2D Mesh for hybrid data and tensor parallelism
def example_2d_mesh(rank, world_size):
    """
    Create a 2D mesh for hybrid parallelism with:
    - 2 data-parallel groups (rows)
    - 2 tensor-parallel groups (columns)
    Total 4 processes in a 2x2 grid
    """
    # Initialize process group if not already done
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size)

    # Create a 2D mesh layout (2x2)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_shape = (2, 2)  # 2 rows, 2 columns

    # Convert the flat process rank to 2D mesh coordinates
    mesh_2d = torch.arange(world_size).reshape(*mesh_shape)
    device_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type, mesh_2d)

    print(f"Rank {rank}: Created 2D device mesh: {device_mesh}")
    return device_mesh


# Example 2: Multiple meshes for different model components
def example_multiple_meshes(rank, world_size):
    """
    Create multiple meshes to parallelize different parts of a model differently:
    - First mesh for encoder: 2 processes tensor parallel
    - Second mesh for decoder: 2 processes tensor parallel
    - Different components could scale differently this way
    """
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Assuming 4 processes, split them for different parts
    encoder_ranks = torch.tensor([0, 1])
    decoder_ranks = torch.tensor([2, 3])

    # Create the meshes
    encoder_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type, encoder_ranks.reshape(2))
    decoder_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type, decoder_ranks.reshape(2))

    if rank < 2:
        print(f"Rank {rank}: Part of encoder mesh: {encoder_mesh}")
    else:
        print(f"Rank {rank}: Part of decoder mesh: {decoder_mesh}")

    # Return the mesh this rank belongs to
    return encoder_mesh if rank < 2 else decoder_mesh


# Example 3: 3D Mesh for complex parallelism strategies
def example_3d_mesh(rank, world_size):
    """
    Create a 3D mesh layout for complex parallelism:
    - Dimension 0: Pipeline parallelism
    - Dimension 1: Tensor parallelism
    - Dimension 2: Data parallelism
    Total 8 processes in a 2x2x2 grid
    """
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_shape = (2, 2, 2)  # 2x2x2 3D mesh

    # Convert flat rank to 3D mesh coordinates
    mesh_3d = torch.arange(world_size).reshape(*mesh_shape)
    device_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type, mesh_3d)

    print(f"Rank {rank}: Created 3D device mesh: {device_mesh}")
    # Get mesh coordinates for this rank
    coords = [i.item() for i in torch.where(mesh_3d == rank)]
    print(f"Rank {rank}: Mesh coordinates: {coords}")

    return device_mesh


# Example 4: Using device mesh with tensor parallelism for transformers
class SimpleTransformerBlock(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=2048, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


def example_tensor_parallel_with_mesh(rank, world_size):
    """
    Use DeviceMesh to enable tensor parallelism for a transformer block
    """
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # For tensor parallelism, we'll use a 1D mesh across all processes
    mesh_shape = (world_size, )
    device_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type,
        torch.arange(world_size).reshape(*mesh_shape))

    print(f"Rank {rank}: Created tensor parallel mesh: {device_mesh}")

    # Create a transformer block
    model = SimpleTransformerBlock()

    # Parallelize the model using the device mesh
    # The first Linear in the FF gets ColwiseParallel (split activations)
    # The second Linear in the FF gets RowwiseParallel (split weights)
    tp_model = parallelize_module(module=model,
                                  device_mesh=device_mesh,
                                  parallelize_plan={
                                      "attention.in_proj_weight":
                                      ColwiseParallel(),
                                      "attention.out_proj.weight":
                                      RowwiseParallel(),
                                      "ff.0.weight":
                                      ColwiseParallel(),
                                      "ff.2.weight":
                                      RowwiseParallel(),
                                  })

    print(f"Rank {rank}: Model parallelized with tensor parallelism")
    return tp_model, device_mesh


# Example 5: Model FSDP with nested tensor parallelism using device mesh
class NestedParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = SimpleTransformerBlock(512, 2048, 8)
        self.decoder = SimpleTransformerBlock(512, 2048, 8)
        self.output = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.output(x.mean(dim=1))


def example_nested_parallelism(rank, world_size):
    """
    Using device mesh for nested parallelism:
    - Data parallelism (FSDP) at the outer level
    - Tensor parallelism for specific layers
    """
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # For a total of 4 processes, create a 2x2 mesh
    mesh_shape = (2, 2)  # 2 data parallel, 2 tensor parallel
    mesh_2d = torch.arange(world_size).reshape(*mesh_shape)
    device_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type, mesh_2d)

    # Get submeshes for different parallel dimensions
    # First dimension for data parallelism
    dp_mesh = device_mesh.get_dim_mesh(dims=[0])
    # Second dimension for tensor parallelism
    tp_mesh = device_mesh.get_dim_mesh(dims=[1])

    print(f"Rank {rank}: Full mesh: {device_mesh}")
    print(f"Rank {rank}: DP submesh: {dp_mesh}")
    print(f"Rank {rank}: TP submesh: {tp_mesh}")

    # Create model
    model = NestedParallelModel()

    # Apply tensor parallelism to transformer blocks
    for name, submodule in model.named_children():
        if isinstance(submodule, SimpleTransformerBlock):
            parallelize_module(module=submodule,
                               device_mesh=tp_mesh,
                               parallelize_plan={
                                   "attention.in_proj_weight":
                                   ColwiseParallel(),
                                   "attention.out_proj.weight":
                                   RowwiseParallel(),
                                   "ff.0.weight": ColwiseParallel(),
                                   "ff.2.weight": RowwiseParallel(),
                               })

    # Then wrap with FSDP using the data parallel dimension
    fsdp_model = FSDP(
        model,
        device_mesh=dp_mesh,
        cpu_offload=CPUOffload(offload_params=True),
    )

    print(f"Rank {rank}: Applied nested parallelism")
    return fsdp_model, device_mesh


# Example 6: Using device mesh with pipeline parallelism
def example_pipeline_parallel_mesh(rank, world_size):
    """
    Use device mesh to set up pipeline parallelism across GPUs
    """
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size)

    # For pipeline parallelism we'll use a 1D mesh
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    pp_mesh = torch.distributed.device_mesh.DeviceMesh(
        device_type, torch.arange(world_size))

    print(f"Rank {rank}: Created pipeline parallel mesh: {pp_mesh}")

    # In pipeline parallelism, each rank would handle specific layers
    # For demonstration, assign different layers to different ranks
    if rank == 0:
        # First stage: embedding and first transformer block
        model_stage = nn.Sequential(nn.Embedding(10000, 512),
                                    SimpleTransformerBlock(512, 2048, 8))
    elif rank == world_size - 1:
        # Last stage: last transformer block and output
        model_stage = nn.Sequential(SimpleTransformerBlock(512, 2048, 8),
                                    nn.Linear(512, 1000))
    else:
        # Middle stages: transformer blocks
        model_stage = SimpleTransformerBlock(512, 2048, 8)

    print(f"Rank {rank}: Created pipeline stage")

    # Note: Full pipeline parallelism implementation would require additional
    # code for microbatching and synchronization between stages

    return model_stage, pp_mesh


# Function to run on each process
def run_example(rank, world_size, example_num):
    if example_num == 1:
        mesh = example_2d_mesh(rank, world_size)
    elif example_num == 2:
        mesh = example_multiple_meshes(rank, world_size)
    elif example_num == 3:
        mesh = example_3d_mesh(rank, world_size)
    elif example_num == 4:
        model, mesh = example_tensor_parallel_with_mesh(rank, world_size)
    elif example_num == 5:
        model, mesh = example_nested_parallelism(rank, world_size)
    elif example_num == 6:
        model, mesh = example_pipeline_parallel_mesh(rank, world_size)

    # Cleanup
    dist.destroy_process_group()


# Usage example
if __name__ == "__main__":
    import torch.multiprocessing as mp
    args = parse_args()

    # Example to run (choose from 1-6)
    example_to_run = args.e

    # Number of processes needed for each example
    processes_needed = {1: 4, 2: 4, 3: 8, 4: 4, 5: 4, 6: 4}

    # Spawn processes
    mp.spawn(run_example,
             args=(processes_needed[example_to_run], example_to_run),
             nprocs=processes_needed[example_to_run])
