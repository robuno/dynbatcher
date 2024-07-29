import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np


def load_mnist(merge_trainset=False, batch_sizes_train=None, batch_size_test=64, ratios=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if merge_trainset:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        # Split the dataset
        subsets = split_dataset(dataset=trainset, 
                                ratios=ratios,
                                batch_sizes=batch_sizes_train)

        # Create DataLoaders for each subset
        dataloaders = create_dataloaders(subsets, batch_sizes_train)

        # Merge the DataLoaders
        trainloader = CombinedDataLoader(dataloaders)

        # Example usage of the merged DataLoader
        num_train_batches = len(trainloader)  
        print(f"Total number of batches in merged DataLoader: {num_train_batches}")

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        return trainloader, testloader

    else:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size_test, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        return trainloader, testloader


def create_split_indices(total_samples, ratios=None, batch_sizes=None):
    # Your code for create_split_indices function
    if batch_sizes is None:
        raise ValueError("batch_sizes must be provided")

    num_splits = len(batch_sizes)

    if ratios is None:
        # Create equal ratios if not provided
        ratio = 1.0 / num_splits
        ratios = [ratio] * num_splits
    elif len(ratios) != num_splits:
        raise ValueError("The number of ratios must match the number of batch sizes")

    if abs(sum(ratios) - 1.0) > 1e-6:  # Using a small epsilon for float comparison
        raise ValueError("The sum of ratios must be approximately 1.0")

    indices = []
    current_index = 0

    for ratio in ratios[:-1]:  # We don't need to process the last ratio
        split_size = int(total_samples * ratio)
        current_index += split_size
        indices.append(current_index)

    return indices

def split_dataset(dataset, ratios=None, batch_sizes=None):
    # Your code for split_dataset function
    total_samples = len(dataset)
    split_indices = create_split_indices(total_samples, ratios, batch_sizes)
    split_indices = sorted(split_indices)  # Ensure the split indices are sorted
    subsets = []
    start_idx = 0

    for end_idx in split_indices:
        subset = Subset(dataset, range(start_idx, end_idx))
        subsets.append(subset)
        start_idx = end_idx

    # Append the remaining part of the dataset as the last subset
    if start_idx < total_samples:
        subset = Subset(dataset, range(start_idx, total_samples))
        subsets.append(subset)

    return subsets

def create_dataloaders(subsets, batch_sizes):
    # Your code for create_dataloaders function
    dataloaders = []
    for subset, batch_size in zip(subsets, batch_sizes):
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)
    return dataloaders



class CombinedDataLoader:
    """
    Combine multiple DataLoaders into a single DataLoader that iterates over them in sequence.
    """
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.batch_map = []
        batch_start = 0
        for dataloader in dataloaders:
            self.batch_map.append((dataloader, batch_start))
            batch_start += len(dataloader)
        self.total_batches = batch_start

    def __iter__(self):
        for dataloader in self.dataloaders:
            for batch in dataloader:
                yield batch

    def __len__(self):
        return self.total_batches

    def get_batch_by_index(self, index):
        """
        Get the batch by index.
        
        Args:
            index (int): The index of the batch to retrieve.
        
        Returns:
            batch: The batch at the specified index.
        """
        if index >= self.total_batches or index < 0:
            raise IndexError(f"Index {index} out of range for CombinedDataLoader with {self.total_batches} batches.")
        
        for dataloader, batch_start in self.batch_map:
            if index < batch_start + len(dataloader):
                batch_index = index - batch_start
                for i, batch in enumerate(dataloader):
                    if i == batch_index:
                        return batch

        raise RuntimeError("Batch index mapping error in CombinedDataLoader.")



