import io
import time
import os
import torch
import torch.distributed as dist
import torch.nn as nn

HIDDEN = 256

GLOO_GROUP = None

# TLDR: model in rank 0-7 send to model in rank 8-15

class SimpleLayer(nn.Module):
    def __init__(self, layer_id):
        super(SimpleLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU()
        )
        self.layer_id = layer_id
        
    def forward(self, x):
        return self.layer(x)


class DistributedModel(nn.Module):
    def __init__(self, world_size, local_rank):
        super(DistributedModel, self).__init__()
        self.world_size = world_size
        self.rank = dist.get_rank() 
        self.local_rank = local_rank

        self.layer = SimpleLayer(self.rank)
        self.layer.to(f'cuda:{self.local_rank}')
        
    def forward(self, x):
        global GLOO_GROUP
        output = self.layer(x)

        # NODE 1 has NVLINK - so send with nccl backend
        if self.rank < 7:
            next_rank = self.rank+1
            dist.send(output, dst=next_rank)
        elif self.rank == 7:
            next_rank = 0
            dist.send(output, dst=next_rank)
        
        # 
        # NODE2 use gloo
        # 
        elif self.rank < 15:
            next_rank = self.rank+1
            dist.send(output.cpu(), dst=next_rank, group=GLOO_GROUP)
        
        elif self.rank == 15:
            next_rank = 8
            dist.send(output.cpu(), dst=next_rank, group=GLOO_GROUP)
        else:
            raise RuntimeError(f'{self.rank=}')
        return output
    
    def recv_layer(self):
        global GLOO_GROUP
        assert 7 < self.rank < 16, f'{self.rank=} try recv'
        peer = self.rank-8

        # NOTE: cpu send has no prob

        ## cpu send
        self.layer.cpu()
        for param in self.layer.parameters():
            dist.recv(param.data.cpu(), src=peer, group=GLOO_GROUP)
        self.layer.to(f'cuda:{self.local_rank}')

        ## gpu send
        #for param in self.layer.parameters():
        #    dist.recv(param.data, src=peer)


    
    def send_layer(self):
        global GLOO_GROUP
        assert self.rank < 8, f'{self.rank=} try sends'
        peer = self.rank + 8

        # NOTE: cpu send has no prob

        ## cpu send
        self.layer.cpu()
        for param in self.layer.parameters():
            dist.send(param.data.cpu(), dst=peer, group=GLOO_GROUP)
        self.layer.to(f'cuda:{self.local_rank}')


        ## gpu send
        #for param in self.layer.parameters():
        #    dist.send(param.data, dst=peer)


def setup():
    global GLOO_GROUP
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # GLOO_GROUP
    ranks = [i for i in range(16)]
    GLOO_GROUP = dist.new_group(ranks=ranks, backend='gloo')
    return local_rank


def main():
    global GLOO_GROUP
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[SETUP] on rank {rank} (local_rank: {local_rank}), world_size: {world_size}, {torch.cuda.device_count()=} {torch.cuda.current_device()=}")
    assert world_size == 16, f"{world_size=}"
    
    # Create the distributed model
    model = DistributedModel(world_size, local_rank)
    
    # Generate input on rank 0 only
    if rank == 0:
        # Create random input
        input_tensor = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        
        with torch.no_grad():
            output = model.forward(input_tensor)
        
        # # If rank 0, receive the final result from last rank
        this_out = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        dist.recv(this_out, src=7)
        print(f"Rank 0: Received final output from rank 7")

        print(f"Rank 0: send to external")
        dist.send(input_tensor, dst=8)

        t1 = time.time()
        model.send_layer()
        t2 = time.time()
        print(f'Rank 0: send {t2-t1:.2f}s')
        
        # # Run validation
        extern_out = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        dist.recv(extern_out, src=8)
        #extern_out = torch.zeros(4, HIDDEN, device=f'cpu')
        #dist.recv(extern_out, src=8, group=GLOO_GROUP)
        
        # # Compare results
        print("Rank 0: Running validation")
        diff = torch.max(torch.abs(this_out - extern_out))
        print(f"Maximum difference between distributed and local execution: {diff.item()}")
        if diff < 1e-5:
            print("Validation successful!")
        else:
            print("Validation failed! Results don't match.")
            
    elif rank < 8:
        input_tensor = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        dist.recv(input_tensor, src=rank - 1)
        
        ## foward and send
        with torch.no_grad():
           output = model.forward(input_tensor)

        t1 = time.time()
        model.send_layer()
        t2 = time.time()
        print(f'{rank=}: send {t2-t1:.2f}s')
    
    ####################################################
    # NODE 2
    ####################################################
    elif rank == 8:
        # recv input
        input_tensor = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        dist.recv(input_tensor, src=0)

        ## recv model
        model.recv_layer()

        # forward
        with torch.no_grad():
            output = model.forward(input_tensor)

        # receive
        out = torch.zeros(4, HIDDEN, device=f'cpu')
        dist.recv(out, src=15, group=GLOO_GROUP)
        print(f"Rank 8: Received final output from rank 15")

        # send to 0
        out = out.to(f'cuda:{local_rank}')
        dist.send(out, dst=0)

    elif rank < 16:
        # recv model
        model.recv_layer()

        # last input and forward
        #input_tensor = torch.zeros(4, HIDDEN, device=f'cuda:{local_rank}')
        input_tensor = torch.zeros(4, HIDDEN, )
        dist.recv(input_tensor, src=rank - 1,group=GLOO_GROUP)

        input_tensor = input_tensor.to(f'cuda:{local_rank}')
        with torch.no_grad():
            output = model.forward(input_tensor)
    else:
        raise RuntimeError(f'{rank=}')

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()