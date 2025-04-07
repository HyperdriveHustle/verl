import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
import time


# Create a simple dataset
class RandomDataset(Dataset):

    def __init__(self, size=100):
        self.data = torch.randn(size, 10)  # 100 samples, each with 10 features
        self.labels = torch.randint(0, 2, (size, ))  # Binary labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def demo_dataset():
    # Create the dataset
    dataset = RandomDataset()

    # Direct access to dataset items
    print("Direct access to dataset:")
    sample_idx = 5
    features, label = dataset[sample_idx]
    print(f"Sample {sample_idx}:")
    print(f"Features shape: {features.shape}")
    print(f"Label: {label}")

    # Iterate through the dataset directly
    print("\nIterating through dataset directly:")
    for i, (features, label) in enumerate(dataset):
        if i < 3:  # Just show the first 3 samples
            print(f"Sample {i}:")
            print(f"Features shape: {features.shape}")
            print(f"Label: {label}")
        else:
            break

    # Using with standard PyTorch DataLoader
    print("\nUsing with standard DataLoader:")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for i, batch in enumerate(dataloader):
        features, labels = batch
        if i == 0:
            print(f"Batch size: {len(features)}")
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
        if i >= 2:  # Only show first few batches
            break

    # Get dataset statistics
    print("\nDataset statistics:")
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimension: {dataset.data.shape[1]}")
    print(f"Label distribution: {torch.bincount(dataset.labels)}")


# Create a simple dataset
class SimpleDataset(Dataset):

    def __init__(self, size=100, feature_dim=10, num_classes=3):
        self.data = torch.randn(size, feature_dim)
        # Create imbalanced classes for demonstration
        weights = torch.tensor([0.6, 0.3, 0.1])
        self.labels = torch.multinomial(weights.repeat(size, 1), 1).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def demo_dataloader():
    dataset = SimpleDataset(size=1000)

    print("1. Basic DataLoader Usage")
    basic_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(basic_loader))
    features, labels = batch
    print(f"Batch features shape: {features.shape}")
    print(f"Batch labels shape: {labels.shape}")

    # Class distribution in the dataset
    class_counts = torch.bincount(dataset.labels)
    print(f"Class distribution in dataset: {class_counts}")

    print("\n2. DataLoader with Weighted Sampling (handling imbalanced data)")
    # Create weights for samples based on class frequencies
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[dataset.labels]
    weighted_sampler = WeightedRandomSampler(weights=sample_weights,
                                             num_samples=len(sample_weights),
                                             replacement=True)

    weighted_loader = DataLoader(dataset,
                                 batch_size=32,
                                 sampler=weighted_sampler)

    # Check class distribution in weighted loader
    weighted_classes = []
    for _, labels in weighted_loader:
        weighted_classes.extend(labels.tolist())

    weighted_class_counts = torch.bincount(torch.tensor(weighted_classes))
    print(
        f"Class distribution with weighted sampling: {weighted_class_counts}")

    print("\n3. DataLoader with Batch Sampler")
    batch_sampler = BatchSampler(torch.randperm(len(dataset)).tolist(),
                                 batch_size=32,
                                 drop_last=False)

    batch_sampler_loader = DataLoader(dataset, batch_sampler=batch_sampler)

    # Iterating through the first few batches
    print("First 3 batch sizes from batch sampler:")
    for i, (features, labels) in enumerate(batch_sampler_loader):
        if i < 3:
            print(f"Batch {i} size: {features.shape[0]}")
        else:
            break

    # print("\n4. DataLoader with num_workers (parallel data loading)")
    # # Time comparison for different num_workers
    # times = []
    # worker_options = [0, 2, 4]

    # for num_workers in worker_options:
    #     start_time = time.time()
    #
    #     parallel_loader = DataLoader(
    #         dataset,
    #         batch_size=32,
    #         shuffle=True,
    #         num_workers=num_workers
    #     )
    #
    #     # Load all data
    #     for _ in parallel_loader:
    #         pass
    #
    #     elapsed = time.time() - start_time
    #     times.append(elapsed)
    #     print(f"Time with {num_workers} workers: {elapsed:.4f} seconds")

    # print("\n5. DataLoader with pinned memory (faster GPU transfer)")
    # pinned_loader = DataLoader(
    #     dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     pin_memory=True  # Speeds up host to GPU transfers
    # )

    # Demo batching behavior
    print("\n6. DataLoader with drop_last option")
    # Create a dataset with size not divisible by batch_size
    small_dataset = SimpleDataset(size=95)
    print(f"Dataset size: {len(small_dataset)}")

    # With drop_last=False (default)
    keep_last_loader = DataLoader(small_dataset,
                                  batch_size=32,
                                  drop_last=False)
    print("With drop_last=False:")
    batch_sizes = [batch[0].shape[0] for batch in keep_last_loader]
    print(f"Batch sizes: {batch_sizes}")

    # With drop_last=True
    drop_last_loader = DataLoader(small_dataset, batch_size=32, drop_last=True)
    print("With drop_last=True:")
    batch_sizes = [batch[0].shape[0] for batch in drop_last_loader]
    print(f"Batch sizes: {batch_sizes}")

    print("\n7. Using collate_fn for custom batching")

    def custom_collate(batch):
        # Separate features and labels
        features = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])

        # Add a custom field - feature norm
        norms = torch.norm(features, dim=1)

        return features, labels, norms

    custom_loader = DataLoader(dataset,
                               batch_size=32,
                               collate_fn=custom_collate)

    # Get the first batch with custom collate function
    features, labels, norms = next(iter(custom_loader))
    print(
        f"Custom batch: features {features.shape}, labels {labels.shape}, norms {norms.shape}"
    )


def main():
    demo_dataset()
    print('*' * 100)
    print('*' * 100)
    demo_dataloader()


if __name__ == "__main__":
    main()
