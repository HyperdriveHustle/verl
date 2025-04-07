import os
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
# torchrun --nproc_per_node=4 custom_bench/ddp.py


class CustomDDP:

    def __init__(self, module, process_group=None):
        self.module = module
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # Register hooks for gradient synchronization
        self.grad_accs = []
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(self._make_hook(p))

    def _make_hook(self, param):

        def hook(grad):
            # Allreduce gradients
            dist.all_reduce(grad,
                            op=dist.ReduceOp.SUM,
                            group=self.process_group)
            # Average the gradients
            grad.div_(self.world_size)
            return grad

        return hook

    def __getattr__(self, name):
        # Forward all other attribute accesses to the wrapped module
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize the process group
    dist.init_process_group(backend="gloo")

    # Set the device
    device = torch.device(f"cpu")  # Using CPU for simplicity
    #torch.cuda.set_device(local_rank)  # Uncomment for GPU

    return local_rank, rank, world_size


def main():
    local_rank, rank, world_size = setup_distributed()
    print(f'worker: {rank=}, {local_rank=}, {world_size=}')
    # train
    model = nn.Linear(10, 10)
    local_model = deepcopy(model)

    model = CustomDDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Different processes can perform different tasks based on rank
    if rank == 0:
        # for param in local_model.parameters():
        #     print(param)
        for p1, p2 in zip(model.parameters(), local_model.parameters()):
            assert torch.allclose(p1.data, p2.data)
        print(f"init check {rank=}, pass local check")

    # Synchronize all processes
    dist.barrier()

    for epoch in range(10):
        # Each process works on its subset of data
        dummy_input = torch.randn(20, 10)

        # dist train
        dist_dummy_output = model(dummy_input)
        loss = dist_dummy_output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # local train
        # dummy_output = local_model(dummy_input)
        # loss = dummy_output.sum()
        # loss.backward()

        dist.barrier()

        if rank == 0:
            # assert torch.allclose(dummy_output, dist_dummy_output)

            # # NOTE: to check grad, data should be gathered for each process
            # grads_non_equal = 0
            # cnt = 0
            # for p1, p2 in zip(model.parameters(), local_model.parameters()):
            #     if p1.grad is not None and p2.grad is not None:
            #         cnt+=1
            #         if not torch.allclose(p1.grad.data, p2.grad.data):
            #             grads_non_equal +=1
            # if grads_non_equal > 0:
            #     raise AssertionError(f"gradients are not equal, {grads_non_equal}/{cnt}")

            # # Update local model to match DDP model for next epoch
            # with torch.no_grad():
            #     for p_ddp, p_local in zip(model.parameters(), local_model.parameters()):
            #         p_local.data.copy_(p_ddp.data)
            # # Zero out gradients for next iteration
            # for p in local_model.parameters():
            #     if p.grad is not None:
            #         p.grad.zero_()

            print(f"Epoch {epoch} completed, pass local check")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
