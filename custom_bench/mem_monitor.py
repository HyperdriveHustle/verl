import torch
import gc
import contextlib

class GPUMemoryMonitor:
    def __init__(self, device=None):
        """
        Initialize GPU Memory Monitor
        
        Args:
            device (torch.device, optional): GPU device to monitor. 
                               Defaults to current CUDA device if available.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_memory = 0
        self.peak_memory = 0
        
    def __enter__(self):
        """
        Context manager entry point
        Capture initial memory state
        """
        # Clear cache and collect garbage before measuring
        torch.cuda.empty_cache()
        gc.collect()
        
        # Record initial memory allocation
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated(self.device)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point
        Calculate and print memory usage statistics
        """
        if torch.cuda.is_available():
            # Get final memory allocation
            final_memory = torch.cuda.memory_allocated(self.device)
            
            # Calculate memory delta
            memory_delta = final_memory - self.initial_memory
            
            # Get peak memory usage
            self.peak_memory = torch.cuda.max_memory_allocated(self.device)
            
            # Print memory usage statistics
            print(f"\nGPU Memory Stats for Device {self.device}:")
            print(f"Initial Memory: {self.initial_memory / 1024**2:.2f} MB")
            print(f"Final Memory: {final_memory / 1024**2:.2f} MB")
            print(f"Memory Delta: {memory_delta / 1024**2:.2f} MB")
            print(f"Peak Memory Usage: {self.peak_memory / 1024**2:.2f} MB")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        return False  # Propagate any exceptions
    
    def get_peak_memory(self):
        """
        Get peak memory usage in megabytes
        
        Returns:
            float: Peak memory usage in MB
        """
        return self.peak_memory / 1024**2 if torch.cuda.is_available() else 0

# Example usage demonstration
def train_model():
    # Simulated model and training setup
    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Generate some random input data
    input_data = torch.randn(32, 100).cuda()
    target = torch.randn(32, 10).cuda()
    
    # Use the GPU Memory Monitor during a training step
    with GPUMemoryMonitor() as mem_monitor:
        # Simulate a training step
        optimizer.zero_grad()
        output = model(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    # Optional: Get peak memory usage
    peak_memory = mem_monitor.get_peak_memory()
    print(f"\nPeak Memory Usage: {peak_memory:.2f} MB")

# Demonstrate usage
if __name__ == "__main__":
    # Ensure CUDA is available
    if torch.cuda.is_available():
        train_model()
    else:
        print("CUDA is not available. Cannot run GPU memory monitoring.")