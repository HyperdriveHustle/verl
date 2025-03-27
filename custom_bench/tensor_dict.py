import torch
import tensordict
from tensordict import TensorDict
import sys

def analyze_tensordict(td):
    """
    Comprehensive analysis of a TensorDict's size and memory consumption
    
    Parameters:
    -----------
    td : TensorDict
        The TensorDict to be analyzed
    
    Returns:
    --------
    dict : A dictionary containing various size and memory metrics
    """
    # Basic size information
    print("TensorDict Analysis:")
    print("-" * 20)
    
    # 1. Shape of the TensorDict
    print(f"Shape: {td.shape}")
    
    # 2. Detailed batch size
    print(f"Batch Size: {td.batch_size}")
    
    # 3. Keys in the TensorDict
    print(f"Keys: {list(td.keys())}")
    
    # 4. Memory consumption for each tensor
    total_memory = 0
    print("\nMemory Breakdown:")
    for key, tensor in td.items():
        # Estimate memory size in bytes
        tensor_memory = tensor.element_size() * tensor.nelement()
        total_memory += tensor_memory
        
        print(f"{key}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Memory: {tensor_memory / (1024 * 1024):.2f} MB")
    
    # 5. Total memory consumption
    print(f"\nTotal Memory Consumption: {total_memory / (1024 * 1024):.2f} MB")
    
    # 6. Detailed size information using sys
    print(f"\nPython Object Size: {sys.getsizeof(td)} bytes")
    
    return {
        'shape': td.shape,
        'batch_size': td.batch_size,
        'keys': list(td.keys()),
        'total_memory_mb': total_memory / (1024 * 1024)
    }

def main():
    # Example usage
    # Create a sample TensorDict
    example_td = TensorDict({
        'observations': torch.randn(10, 3, 64, 64),  # Image-like observations
        'actions': torch.randn(10, 4),               # Action tensor
        'rewards': torch.randn(10)                   # Reward tensor
    }, batch_size=10)

    # Analyze the TensorDict
    analysis_results = analyze_tensordict(example_td)

    # demo
    a = torch.rand(3, 4)
    b = torch.rand(3, 4, 5)
    tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])

    # reshape
    reshaped_tensordict = tensordict.reshape(-1)
    assert reshaped_tensordict.batch_size == torch.Size([12])
    assert reshaped_tensordict["a"].shape == torch.Size([12])
    assert reshaped_tensordict["b"].shape == torch.Size([12, 5])

    # split 
    chunks = tensordict.split([3, 1], dim=1)
    assert chunks[0].batch_size == torch.Size([3, 3])
    #print(f'chunk0 {chunks.}')
    assert chunks[1].batch_size == torch.Size([3, 1])
    torch.testing.assert_close(chunks[0]["a"], tensordict["a"][:, :-1])

    chunks = tensordict.split(2)
    for idx, chunk in enumerate(chunks):
        print(f'*'*100 + f' {idx}')
        print(type(chunk), chunk.batch_size)
        #print(chunk.keys())
        for key, tensor in chunk.items():
            print(key)
            print(tensor.shape)

    print(f'='*100)
    a = torch.rand(3, 4)
    b = torch.rand(3, 5)
    tensordict = TensorDict({"a": a, "b": b}, batch_size=3)
    chunks = tensordict.split(1, dim=0)
    for idx, chunk in enumerate(chunks):
        print(f'*'*100 + f' {idx}')
        print(type(chunk), chunk.batch_size)
        #print(chunk.keys())
        for key, tensor in chunk.items():
            print(key)
            print(tensor.shape)

    # print(f'='*100)
    # chunks = tensordict.split(0, dim=0) # XXX: results in deadlock
    # for idx, chunk in enumerate(chunks):
    #     print(f'*'*100 + f' {idx}')
    #     print(type(chunk), chunk.batch_size)
    #     #print(chunk.keys())
    #     for key, tensor in chunk.items():
    #         print(key)
    #         print(tensor.shape)




if __name__ == "__main__":
    main()