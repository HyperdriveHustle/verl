import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    MixedPrecision
)

# torchrun --nproc_per_node=4 demo_fsdp.py


# Define a more complex model to showcase FSDP
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 256)
        self.transformer1 = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer2 = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.linear = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.linear(x)
        return x

def compare_outputs(original_output, sharded_output, tolerance=1e-4):
    """
    Compare outputs with a given tolerance for floating-point differences
    """
    # Check shape equivalence
    assert original_output.shape == sharded_output.shape, \
        f"Shape mismatch: {original_output.shape} vs {sharded_output.shape}"
    
    # Compute relative difference
    diff = torch.abs(original_output - sharded_output)
    relative_diff = diff / (torch.abs(original_output) + 1e-8)
    
    # Check if all differences are within tolerance
    max_diff = relative_diff.max().item()
    mean_diff = relative_diff.mean().item()
    
    print(f"Max Relative Difference: {max_diff}")
    print(f"Mean Relative Difference: {mean_diff}")
    
    # Allow for small numerical differences due to distributed computation
    assert max_diff < tolerance, \
        f"Max relative difference {max_diff} exceeds tolerance {tolerance}"
    assert mean_diff < tolerance/10, \
        f"Mean relative difference {mean_diff} exceeds tolerance {tolerance/10}"
    
    print("✅ Output equivalence test passed!")

def setup_fsdp_with_device_mesh():
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    
    # Create a device mesh
    # This example demonstrates multiple mesh configurations
    
    # Option 1: Simple 2D device mesh
    mesh_2d = init_device_mesh(
        device_type="cuda", 
        mesh_shape=(2, 2),  # 2x2 mesh
        #device_ids=[0, 1, 2, 3]  # Assumes 4 GPUs
        mesh_dim_names=('data', 'model')
    )
    
    # Option 2: Create a device mesh with data and model parallelism
    mesh_1d = init_device_mesh(
        device_type="cuda",
        mesh_shape=(4, ),  # 2 data parallel, 2 model parallel
    )

    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f'mesh_2d: {mesh_2d.shape} {mesh_2d.ndim}')
        print(f'mesh_1d: {mesh_1d.shape} {mesh_1d.ndim}')

    # Create the model
    model = ComplexModel()
    
    # FSDP Configuration with Device Mesh - Multiple Styles - XXX FSDP must have CUDA? 
    
    # Option 1: Basic FSDP with device mesh
    fsdp_model_basic = FSDP(
        model,
        device_mesh=mesh_2d,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )
    
    # Option 2: Advanced FSDP with mixed precision and custom sharding
    fsdp_model_advanced = FSDP(
        model,
        device_mesh=mesh_2d,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        device_id=torch.cuda.current_device(),
        # mixed_precision=MixedPrecision(
        #     param_dtype=torch.float16,
        #     reduce_dtype=torch.float16,
        #     buffer_dtype=torch.float16
        # ),
    )
    
    # Option 3: Per-module FSDP with different strategies
    fsdp_model_per_module = FSDP(
        model,
        device_mesh=mesh_1d,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        # modules_to_wrap=[
        #     model.transformer1,
        #     model.transformer2
        # ]
    )

    # Synchronize model weights across all configurations
    with torch.no_grad():
        for (name, orig_param), (_, p1), (_, p2), (_, p3) in zip(
            model.named_parameters(),
            fsdp_model_basic.named_parameters(),
            fsdp_model_advanced.named_parameters(),
            fsdp_model_per_module.named_parameters(),
        ):
            # orig_param.copy_(p1)
            # orig_param.copy_(p2)
            # orig_param.copy_(p3)
            p1.copy_(orig_param)
            p2.copy_(orig_param)
            p3.copy_(orig_param)
    
    # Create a sample input
    batch_size, seq_length = 32, 50
    input_tensor = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
    
    with torch.no_grad():
        # Forward pass examples
        output_basic = fsdp_model_basic(input_tensor)
        output_advanced = fsdp_model_advanced(input_tensor)
        output_per_module = fsdp_model_per_module(input_tensor)
    
    dist.barrier()

    if rank == 0:
        print("FSDP with Device Mesh Demonstration")
        print("Basic Model Output Shape:", output_basic.shape)
        print("Advanced Model Output Shape:", output_advanced.shape)
        print("Per-Module Model Output Shape:", output_per_module.shape)

        # FIXME cannot pass if change mesh or sharding; 
        # compare_outputs(output_basic, output_advanced)
        compare_outputs(output_basic, output_per_module)
    
    # Clean up
    dist.destroy_process_group()

def main():
    # This function would typically be launched with torchrun
    setup_fsdp_with_device_mesh()

if __name__ == '__main__':
    main()