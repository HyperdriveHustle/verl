import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ExpertLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class DistributedMoE(nn.Module):

    def __init__(self, input_dim, output_dim, num_experts, world_size):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.world_size = world_size

        # Router network (shared across all processes)
        self.router = nn.Linear(input_dim, num_experts)

        # Each process handles a subset of experts
        experts_per_process = num_experts // world_size
        self.local_expert_indices = list(
            range(dist.get_rank() * experts_per_process,
                  (dist.get_rank() + 1) * experts_per_process))

        # Create only the local experts for this process
        self.local_experts = nn.ModuleList([
            ExpertLayer(input_dim, output_dim)
            for _ in range(len(self.local_expert_indices))
        ])

        print(
            f"Process {dist.get_rank()} handles experts {self.local_expert_indices}"
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Get routing probabilities
        routing_logits = self.router(x)
        routing_probs = F.softmax(routing_logits, dim=1)

        # Get top-k routing decisions (using top-1 for simplicity)
        _, indices = torch.topk(routing_probs, k=1, dim=1)
        flat_idx = indices.view(-1)

        if dist.get_rank() == 0:
            print(f"Routing decisions: {indices}")
            print(f"flat: {flat_idx}")

        # Create a map from expert index to input indices
        expert_to_inputs = {}
        input_to_expert = {}
        for input_idx, expert_idx in enumerate(flat_idx.tolist()):
            if expert_idx not in expert_to_inputs:
                expert_to_inputs[expert_idx] = []
            expert_to_inputs[expert_idx].append(input_idx)
            input_to_expert[input_idx] = expert_idx

        # Collect local inputs for local experts
        local_outputs = torch.zeros(batch_size,
                                    self.output_dim,
                                    device=x.device)

        # Process inputs for each local expert
        for i, expert_idx in enumerate(self.local_expert_indices):

            # XXX only the token to local expert will be considered
            if expert_idx in expert_to_inputs:

                local_input_indices = expert_to_inputs[expert_idx]
                if local_input_indices:
                    local_inputs = x[local_input_indices]
                    local_expert_outputs = self.local_experts[i](local_inputs)
                    local_outputs[local_input_indices] = local_expert_outputs

        # All-reduce across processes to gather outputs from all experts
        dist.all_reduce(local_outputs, op=dist.ReduceOp.SUM)

        return local_outputs


def init_distributed():
    # Initialize process group
    dist.init_process_group(backend="gloo")
    #torch.cuda.set_device(dist.get_rank())
    print(f"Initialized process {dist.get_rank()} of {dist.get_world_size()}")


def main():
    init_distributed()

    # Model parameters
    input_dim = 256
    output_dim = 256
    num_experts = 8
    world_size = dist.get_world_size()

    # Create model
    model = DistributedMoE(input_dim, output_dim, num_experts, world_size)
    #model = DDP(model, device_ids=[dist.get_rank()])

    # XXX assuming each MoE layer gets the same input
    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(x)
    print(f"Process {dist.get_rank()}: Output shape: {output.shape}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
