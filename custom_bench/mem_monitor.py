import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import ray
import psutil
import GPUtil
import time
import os

class MemoryMonitor:
    """
    A utility class to monitor system and GPU memory usage
    during distributed training.
    """
    @staticmethod
    def get_cpu_memory_usage():
        """
        Get current CPU memory usage.
        
        Returns:
            dict: Memory usage statistics
        """
        process = psutil.Process(os.getpid())
        return {
            'total_memory_mb': psutil.virtual_memory().total / (1024 * 1024),
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'memory_percent': process.memory_percent()
        }
    
    @staticmethod
    def get_gpu_memory_usage():
        """
        Get current GPU memory usage.
        
        Returns:
            list: GPU memory usage for each available GPU
        """
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    'gpu_id': gpu.id,
                    'total_memory_mb': gpu.memoryTotal,
                    'used_memory_mb': gpu.memoryUsed,
                    'free_memory_mb': gpu.memoryFree,
                    'memory_utilization_percent': gpu.memoryUtil * 100
                }
                for gpu in gpus
            ]
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            return []

class DistributedTrainingExample:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train_step(self, data, target):
        """
        Perform a single training step with memory logging.
        
        Args:
            data (torch.Tensor): Input data
            target (torch.Tensor): Target labels
        """
        # Log memory before training step
        print("\n--- Memory Before Training Step ---")
        print("CPU Memory:", MemoryMonitor.get_cpu_memory_usage())
        print("GPU Memory:", MemoryMonitor.get_gpu_memory_usage())
        
        # Standard training step
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        
        # Log memory after training step
        print("\n--- Memory After Training Step ---")
        print("CPU Memory:", MemoryMonitor.get_cpu_memory_usage())
        print("GPU Memory:", MemoryMonitor.get_gpu_memory_usage())

def pytorch_distributed_training(rank, world_size):
    """
    Example of distributed PyTorch training with memory monitoring.
    
    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
    """
    # Initialize distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create a simple model
    model = nn.Linear(10, 5).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create training example with memory monitoring
    trainer = DistributedTrainingExample(model, criterion, optimizer)
    
    # Simulate training
    for epoch in range(3):
        print(f"\n=== Epoch {epoch+1} ===")
        data = torch.randn(32, 10).to(rank)
        target = torch.randn(32, 5).to(rank)
        
        trainer.train_step(data, target)
        
        # Optional: Synchronize processes
        dist.barrier()

def ray_distributed_training():
    """
    Example of distributed Ray training with memory monitoring.
    """
    # Initialize Ray
    ray.init(num_cpus=4, num_gpus=2)
    
    @ray.remote(num_gpus=1)
    def distributed_worker(worker_id):
        """
        Simulated distributed worker with memory logging.
        
        Args:
            worker_id (int): Unique worker identifier
        """
        print(f"\n=== Worker {worker_id} Memory Report ===")
        print("CPU Memory:", MemoryMonitor.get_cpu_memory_usage())
        print("GPU Memory:", MemoryMonitor.get_gpu_memory_usage())
        
        # Simulate some computational work
        time.sleep(2)
        return worker_id
    
    # Launch multiple distributed workers
    worker_refs = [distributed_worker.remote(i) for i in range(2)]
    
    # Wait for all workers to complete
    results = ray.get(worker_refs)
    print("\nDistributed Workers Results:", results)

def main():
    # PyTorch Distributed Training
    print("\n=== PyTorch Distributed Training ===")
    world_size = torch.cuda.device_count()
    mp.spawn(pytorch_distributed_training, args=(world_size,), nprocs=world_size)
    
    # Ray Distributed Training
    print("\n=== Ray Distributed Training ===")
    ray_distributed_training()

if __name__ == '__main__':
    main()