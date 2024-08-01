# dynbatcher - Dynamic Batch Size Dataloader Generator

`dynbatcher` is a Python package designed to facilitate the creation and management of PyTorch DataLoaders with custom batch sizes and ratios. This package is especially useful for training neural networks with dynamic batch sizes. With `dynbatcher` you can divide a dataset into subsets with different batch sizes and turn it into a single Dataloader ready for training. 

## Features

- Split datasets based on custom ratios and batch sizes.
- Create DataLoaders for each subset with different batch sizes.
- Combine multiple DataLoaders into a single DataLoader.
- Plot samples for a selected batch in created Dataloader.

## Installation

Install `dynbatcher` using pip:

```bash
pip install dynbatcher
```

## Usage

### Importing the package

```
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import dynbatcher

# Define batch sizes and ratios
ratios = [0.5, 0.3, 0.2]              # Corresponding ratios for each subset
batch_sizes_train = [32, 64, 128]     # Corresponding batch sizes for each subset

# Add transforms and download datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set dataloaders and use DBS Merger & Generator
trainloader = dynbatcher.load_merged_trainloader(trainset, batch_sizes_train, ratios, print_info=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Example: Access and print information for the selected batch, and display samples
dynbatcher.print_batch_info(trainloader, index=127, display_samples=True)
```

- `batch_sizes_train`: You can choose which batch sizes you want to split the dataset into.
- `ratios`: You can choose the ratio in which the data will be allocated to the batch sizes you choose for the dataset. If you do not specify a ratio, it will allocate an equal number of samples to the given batch sizes.

### Investigating Batch Samples

<div align="center">
    <img src="https://raw.githubusercontent.com/starkslab/starkslab.github.io/main/dbstraining/static/images/batch128.png" width="250" style="vertical-align: top;" alt="Batch Size 128 Samples" />
    <img src="https://raw.githubusercontent.com/starkslab/starkslab.github.io/main/dbstraining/static/images/batch64.png" width="250" style="vertical-align: top;" alt="Batch Size 64 Samples" />
    <img src="https://raw.githubusercontent.com/starkslab/starkslab.github.io/main/dbstraining/static/images/batch32.png" width="250" style="vertical-align: top;" alt="Batch Size 32 Samples" />
</div>

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License
This project is licensed under the MIT License.

## Acknowledgments
Inspired by the need for flexible and efficient DataLoader management in PyTorch.

```
This `README.md` provides an overview of your project, installation instructions, usage examples, a brief description of the key functions, and information on contributing and licensing.
```