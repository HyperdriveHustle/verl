import io
import os
import torch
import torch.distributed as dist
import torch.nn as nn

HIDDEN = 256


class SimpleLayer(nn.Module):

    def __init__(self, layer_id):
        super(SimpleLayer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.layer_id = layer_id

    def forward(self, x):
        print(f"Rank {dist.get_rank()}: Processing layer {self.layer_id}")
        return self.layer(x)


class DistributedModel(nn.Module):

    def __init__(self, world_size, local_rank):
        super(DistributedModel, self).__init__()
        self.world_size = world_size
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = local_rank
        self.layer = SimpleLayer(self.rank)

        # Move to appropriate device if distributed is initialized
        if dist.is_initialized():
            self.layer.to(f'cuda:{self.local_rank}')

    def forward(self, x, local=False):
        if not local:
            # Move input to current GPU
            x = x.to(f'cuda:{self.local_rank}')

            # Process the current layer
            output = self.layer(x)

            # print(x)
            # print(output)

            # Send output to the next rank, or back to rank 0 if this is the last rank
            if self.rank < self.world_size - 1:
                # Send to next rank
                dist.send(output, dst=self.rank + 1)
                print(f"Rank {self.rank}: Sent output to rank {self.rank + 1}")
            else:
                # Last rank - send back to rank 0 for validation
                dist.send(output, dst=0)
                print(f"Rank {self.rank}: Sent final output to rank 0")

            return output
        else:
            # local mode - process all layers sequentially for validation
            x = x.to(f'cuda:{self.local_rank}')
            for layer_idx in range(self.world_size):
                layer = self.get_layer(layer_idx)

                # TODO check weights - the same
                # print(layer_idx)
                # for name, param in layer.named_parameters():
                #     print(f"Parameter Name: {name}")
                #     print(f"Value:\n{param}\n")  # Prints the tensor values
                #if layer_idx == 0:
                #    for name, param in self.layer.named_parameters():
                #        print(f"Parameter Name: {name}")
                #        print(f"Value:\n{param}\n")  # Prints the tensor values
                #    for name, param in layer.named_parameters():
                #        print(f"Parameter Name: {name}")
                #        print(f"Value:\n{param}\n")  # Prints the tensor values
                #
                #    print('more')
                #    print(self.layer(x))

                #print()
                #print(layer_idx)
                #print(x)
                x = layer(x)
                #print(x)
            return x

    def get_layer(self, idx):
        return getattr(self, f"layer_{idx}", None)

    def collect_all_layers(self):
        """
        Collect all layer weights from all ranks to rank 0 for local validation.
        Only called by rank 0.
        """
        if self.rank == 0:
            for layer_idx in range(self.world_size):
                if layer_idx == 0:
                    # Already have layer 0
                    # XXX set externally!
                    continue

                # Create a placeholder layer for each rank
                temp_layer = SimpleLayer(layer_idx).to(
                    f'cuda:{self.local_rank}')

                # Get state dict of the layer from the corresponding rank
                # Receive state dict from other ranks
                # size_tensor = torch.tensor([0], dtype=torch.long)
                # size_tensor = size_tensor.to(f'cuda:{self.local_rank}')
                # dist.recv(size_tensor, src=layer_idx)

                # size = size_tensor.item()
                # byte_tensor = torch.empty(size, dtype=torch.uint8)
                # byte_tensor = byte_tensor.to(f'cuda:{self.local_rank}')
                # dist.recv(byte_tensor, src=layer_idx)

                # bytes_tensor = byte_tensor.cpu().numpy().tobytes()
                # buffer_io = io.BytesIO(bytes_tensor)
                # state_dict = torch.load(buffer_io)

                # # Load state dict into the temporary layer
                # temp_layer.load_state_dict(state_dict)

                # # Store the layer in the model
                # setattr(self, f"layer_{layer_idx}", temp_layer)
                # print(f"Rank 0: Received layer {layer_idx} from rank {layer_idx}")

                for param in temp_layer.parameters():
                    dist.recv(param.data, src=layer_idx)
                setattr(self, f"layer_{layer_idx}", temp_layer)
                print(
                    f"Rank 0: Received layer {layer_idx} from rank {layer_idx}"
                )

    def send_layer_to_rank0(self):
        if self.rank > 0:

            ## XXX io.BytesIO causes error
            # buffer = io.BytesIO()
            # torch.save(self.layer.state_dict(), buffer)
            # buffer.seek(0)
            # byte_tensor = torch.ByteTensor(list(buffer.getvalue()))
            # byte_tensor = byte_tensor.to(f'cuda:{self.local_rank}')
            # size_tensor = torch.tensor([byte_tensor.size(0)], dtype=torch.long)
            # size_tensor = size_tensor.to(f'cuda:{self.local_rank}')

            # dist.send(size_tensor, dst=0)
            # dist.send(byte_tensor, dst=0)
            # print(f"Rank {self.rank}: Sent layer weights to rank 0")

            for param in self.layer.parameters():
                dist.send(param.data, dst=0)


def setup():
    # Initialize the process group with the default backend
    dist.init_process_group(backend="nccl")

    # Set the device based on local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    # Setup the distributed environment
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(
        f"[SETUP] on rank {rank} (local_rank: {local_rank}), world_size: {world_size}"
    )

    # Make sure we have exactly 8 processes
    assert world_size == 8, f"This script requires exactly 8 processes, but got {world_size}"

    # Create the distributed model
    model = DistributedModel(world_size, local_rank)

    # Generate input on rank 0 only
    if rank == 0:
        # Create random input
        input_tensor = torch.rand(4, HIDDEN)

        # Run distributed forward pass
        output = model.forward(input_tensor)

        # If rank 0, receive the final result from last rank
        final_output = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        dist.recv(final_output, src=world_size - 1)
        print(f"Rank 0: Received final output from rank {world_size - 1}")

        # Wait for all processes to reach this point
        for r in range(1, world_size):
            sync_tensor = torch.zeros(1, device=f'cuda:{local_rank}')
            dist.recv(sync_tensor, src=r)

        # Create local model for validation
        print("Rank 0: Creating local model for validation")
        local_model = DistributedModel(world_size, local_rank)

        # Collect weights from all ranks
        # First, synchronize to make sure all ranks are ready to send their weights
        for r in range(1, world_size):
            dist.send(torch.zeros(1, device=f'cuda:{local_rank}'), dst=r)

        # Now collect all weights
        print("Rank 0: Collecting layer weights from all ranks")
        local_model.collect_all_layers()
        setattr(local_model, f"layer_{0}", model.layer)

        dist.barrier()

        # Run validation
        print("Rank 0: Running local validation")
        with torch.no_grad():
            local_output = local_model.forward(input_tensor, local=True)

        # Compare results
        diff = torch.max(torch.abs(final_output - local_output))
        print(
            f"Maximum difference between distributed and local execution: {diff.item()}"
        )
        if diff < 1e-5:
            print(
                "Validation successful! Distributed and local results match.")
        else:
            print("Validation failed! Results don't match.")

    else:
        # Non-zero ranks wait to receive input from previous rank
        input_tensor = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        dist.recv(input_tensor, src=rank - 1)
        print(f"Rank {rank}: Received input from rank {rank - 1}")

        # Process this layer
        output = model.forward(input_tensor)

        # Synchronize with rank 0 before weight collection
        sync_tensor = torch.ones(1, device=f'cuda:{local_rank}')
        dist.send(sync_tensor, dst=0)

        # Wait for rank 0 to be ready to receive weights
        sync_tensor = torch.zeros(1, device=f'cuda:{local_rank}')
        dist.recv(sync_tensor, src=0)

        # Send layer weights to rank 0
        model.send_layer_to_rank0()

        # print(f'{rank=}: print')
        # layer = model.layer
        # for name, param in layer.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Value:\n{param}\n")  # Prints the tensor values
        dist.barrier()

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
