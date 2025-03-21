import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)


# Simple model for demonstration
class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 5))

    def forward(self, x):
        return self.layers(x)


def setup_dist(rank, world_size):
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()


def train(rank, world_size):
    # Initialize distributed environment
    setup_dist(rank, world_size)

    # Create mesh from CPU devices
    mesh_shape = (world_size,)
    # Define mesh axes
    mesh_axes = (0,)

    # Create CPU device mesh
    device_mesh = torch.distributed.device_mesh.DeviceMesh("cpu", torch.arange(world_size).reshape(*mesh_shape))

    print(f"Rank {rank}: Device mesh created: {device_mesh}")

    # Create model
    model = SimpleModel()

    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        device_mesh=device_mesh,
        cpu_offload=CPUOffload(offload_params=True),
    )

    # Create optimizer
    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)

    # Create dummy data
    batch_size = 8
    input_data = torch.randn(batch_size, 10)
    target = torch.randint(0, 5, (batch_size,))

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = fsdp_model(input_data)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Cleanup
    cleanup()


if __name__ == "__main__":
    # Set number of processes (CPU cores)
    world_size = 4

    # Use torch.multiprocessing to start multiple processes
    import torch.multiprocessing as mp
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
