import torch
import torch.distributed as dist


def setup_distributed():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def dispatch_tokens_to_experts(tokens, expert_assignments, num_experts,
                               world_size, rank):
    """
    Dispatch tokens to their assigned experts across all ranks using all-to-all.
    
    Args:
        tokens: Tensor of shape [batch_size, seq_len, hidden_dim]
        expert_assignments: Tensor of shape [batch_size, seq_len] containing expert indices
        num_experts: Total number of experts across all ranks
        
    Returns:
        Tokens received from other ranks for processing by local experts
    """
    batch_size, seq_len, hidden_dim = tokens.shape

    # Assume num_experts is divisible by world_size for simplicity
    experts_per_rank = num_experts // world_size

    # Step 1: Create a list of tokens that need to go to each rank
    tokens_to_ranks = [[] for _ in range(world_size)]
    token_indices = []  # To keep track of original positions

    for b in range(batch_size):
        for s in range(seq_len):
            # Determine which rank handles this expert
            expert_idx = expert_assignments[b, s].item()
            target_rank = expert_idx // experts_per_rank

            # Store the token with its position information
            tokens_to_ranks[target_rank].append(tokens[b, s])
            token_indices.append((b, s, expert_idx))

    if rank == 0:
        #print(f"Token indices: {token_indices[:5]}")
        pass
    dist.barrier()

    # Step 2: Pad and convert lists to tensors
    max_tokens = max(len(tokens) for tokens in tokens_to_ranks)

    if rank == 0:
        for tokens_rank in tokens_to_ranks:
            print(f"Tokens per rank: {len(tokens_rank)}")
        print(f"Max tokens: {max_tokens}")

    # Pad each list to have the same length and convert to tensors
    send_tensors = []
    for r in range(world_size):
        padded = tokens_to_ranks[r]
        if len(padded) < max_tokens:
            # Padding tokens (will be ignored by receiver)
            padding = torch.zeros((max_tokens - len(padded), hidden_dim),
                                  device=tokens.device)
            padded_tensor = torch.cat([
                torch.stack(padded)
                if padded else torch.tensor([], device=tokens.device), padding
            ],
                                      dim=0)
        else:
            padded_tensor = torch.stack(padded)
        send_tensors.append(padded_tensor)

    if rank == 0:
        for send_tensor in send_tensors:
            print(f"Send tensor: {send_tensor.shape}")
        print(len(send_tensors))
    dist.barrier()

    # Step 3: Perform all-to-all communication
    send_tensor = torch.cat(send_tensors, dim=0)

    # Create output tensor for all-to-all
    recv_tensor = torch.zeros_like(send_tensor.T)

    # TODO padding makes sense, but more need to do ,e.g. input split list
    raise
    dist.all_to_all_single(recv_tensor, send_tensor)

    # Step 4: Process tokens for local experts
    # Split received tokens by expert
    local_expert_start = rank * experts_per_rank
    local_expert_end = local_expert_start + experts_per_rank

    # Process the received tokens with local experts
    # (This is a simplified version, in a real implementation
    # you would route to specific expert networks)
    processed_tokens = {}

    # Simulate processing by local experts
    for i, token in enumerate(recv_tensor):
        # In a real implementation, you would:
        # 1. Route to the correct local expert
        # 2. Process the token with that expert
        # 3. Store the result with position information

        # Simple example processing (just adding expert ID to demonstrate)
        expert_id = local_expert_start + (i % experts_per_rank)
        processed_tokens[i] = token * (expert_id + 1)  # Simple transformation

    # Step 5: Send processed tokens back to their source ranks
    # (Real implementation would use another all-to-all here)

    return processed_tokens


# Example usage
def main():
    rank, world_size = setup_distributed()

    # Create sample data
    batch_size, seq_len, hidden_dim = 4, 8, 64
    num_experts = 8  # Total experts across all ranks

    # Sample tokens
    tokens = torch.randn(batch_size, seq_len, hidden_dim)

    # Randomly assign tokens to experts
    expert_assignments = torch.randint(0, num_experts, (batch_size, seq_len))

    # Dispatch tokens to experts
    processed_tokens = dispatch_tokens_to_experts(tokens, expert_assignments,
                                                  num_experts, world_size,
                                                  rank)

    print(f"Rank {rank} processed {len(processed_tokens)} tokens")


if __name__ == "__main__":
    main()
