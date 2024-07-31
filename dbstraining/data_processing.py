from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset

def load_merged_trainloader(trainset, batch_sizes_train=None, ratios=None, print_info=False):
    """
    Splits a dataset into subsets, creates DataLoaders for each subset with specified batch sizes,
    and merges them into a single CombinedDataLoader.

    Parameters:
    trainset (Dataset): The training dataset.
    batch_sizes_train (list of int, optional): Batch sizes for each subset.
    ratios (list of float, optional): Ratios to split the dataset into subsets. Should sum to 1.0.

    Returns:
    CombinedDataLoader: A DataLoader that iterates over all subsets in sequence.
    """

    # Split the dataset
    subsets = split_dataset(dataset=trainset, 
                            ratios=ratios,
                            batch_sizes=batch_sizes_train,
                            print_info=print_info)

    # Create DataLoaders for each subset
    dataloaders = create_dataloaders(subsets, batch_sizes_train, print_info=print_info)

    # Merge the DataLoaders
    trainloader = CombinedDataLoader(dataloaders)

    # Example usage of the merged DataLoader
    num_train_batches = len(trainloader)  
    print(f"Total number of batches in merged DataLoader: {num_train_batches}")

    return trainloader


def create_split_indices(total_samples, ratios=None, batch_sizes=None):
    """
    Creates split indices for dividing a dataset based on specified ratios and batch sizes.

    Parameters:
    total_samples (int): Total number of samples in the dataset.
    ratios (list of float, optional): Ratios for splitting the dataset. Should sum to 1.0.
    batch_sizes (list of int): Batch sizes for each subset.

    Returns:
    list of int: Indices for splitting the dataset.

    Raises:
    ValueError: If batch_sizes is not provided, ratios do not match batch sizes, or ratios do not sum to 1.0.
    """

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


def split_dataset(dataset, ratios=None, batch_sizes=None, print_info = False):
    """
    Splits a dataset into subsets based on provided ratios and batch sizes.

    Parameters:
    dataset (Dataset): The dataset to split.
    ratios (list of float, optional): Ratios for splitting the dataset.
    batch_sizes (list of int, optional): Batch sizes for each subset.

    Returns:
    list of Subset: List of dataset subsets.

    Raises:
    ValueError: If batch_sizes is None.
    ValueError: If the number of ratios does not match the number of batch sizes.
    ValueError: If the sum of ratios does not approximately equal 1.0.
    """

    total_samples = len(dataset)
    split_indices = create_split_indices(total_samples, ratios, batch_sizes)
    split_indices = sorted(split_indices)  # Ensure the split indices are sorted
    subsets = []
    start_idx = 0

    if print_info:
        print("Split indices:",split_indices)


    for end_idx in split_indices:
        subset = Subset(dataset, range(start_idx, end_idx))
        subsets.append(subset)
        start_idx = end_idx

    # Append the remaining part of the dataset as the last subset
    if start_idx < total_samples:
        subset = Subset(dataset, range(start_idx, total_samples))
        subsets.append(subset)

    return subsets


def create_dataloaders(subsets, batch_sizes, print_info=False):
    """
    Creates a list of DataLoader objects for given dataset subsets and their corresponding batch sizes.

    This function takes a list of dataset subsets and a list of batch sizes, and returns a list of 
    DataLoader objects, each configured with the respective subset and batch size. The DataLoader 
    objects are created with shuffling enabled to ensure that the data is randomly shuffled at each 
    epoch.

    Parameters:
    subsets (list of torch.utils.data.Subset): A list of dataset subsets, where each subset is 
                                               a Subset object containing a portion of the original 
                                               dataset.
    batch_sizes (list of int): A list of batch sizes corresponding to each subset. The length of 
                               this list should match the length of the subsets list.

    Returns:
    list of torch.utils.data.DataLoader: A list of DataLoader objects, each configured with the 
                                         respective subset and batch size.

    """
    total_batches = 0
    dataloaders = []
    for subset, batch_size in zip(subsets, batch_sizes):
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)
        num_batches = len(dataloader)
        total_batches += num_batches
        if print_info:
            print(f"Dataloader created with batch size {batch_size}, containing {len(subset)} samples, number of batches: {num_batches}.")

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