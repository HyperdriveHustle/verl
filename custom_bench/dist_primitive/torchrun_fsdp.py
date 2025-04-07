import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)

# torchrun --nproc_per_node=4 fsdp_example.py


# Simple model for demonstration
class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(10, 100), nn.ReLU(),
                                    nn.Linear(100, 100), nn.ReLU(),
                                    nn.Linear(100, 5))

    def forward(self, x):
        return self.layers(x)


def setup_dist():
    """Initialize the distributed environment that's already set up by torchrun."""
    # Initialize process group - this is already done by torchrun
    # but we need to make sure it's initialized properly
    if not dist.is_initialized():
        dist.init_process_group("gloo")

    # Get rank and world_size from environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Rank {rank}/{world_size} (Local rank: {local_rank}) initialized")

    return rank, world_size, local_rank


def train():
    # Initialize the distributed environment
    rank, world_size, local_rank = setup_dist()

    # Create mesh from CPU devices
    mesh_shape = (world_size, )
    # Define mesh axes
    mesh_axes = (0, )

    # Create CPU device mesh
    device_mesh = torch.distributed.device_mesh.DeviceMesh(
        "cpu",
        torch.arange(world_size).reshape(*mesh_shape))

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
    target = torch.randint(0, 5, (batch_size, ))

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

    # No need to explicitly cleanup as torchrun will handle this


if __name__ == "__main__":
    train()
