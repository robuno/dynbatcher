# DBS-training

`dbstraining` is a Python package designed to facilitate the creation and management of PyTorch DataLoaders with custom batch sizes and ratios. This package is especially useful for training neural networks with dynamic batch sizes.

## Features

- Split datasets based on custom ratios and batch sizes.
- Create DataLoaders for each subset.
- Combine multiple DataLoaders into a single DataLoader.

## Installation

Install `dbstraining` using pip:

```bash
pip install dbstraining
```

## Usage

### Importing the package

```
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import dbstraining

# define batch sizes and ratios
ratios = [0.5, 0.3, 0.2]              # Corresponding ratios for each subset
batch_sizes_train = [32, 64, 128]     # Corresponding batch sizes for each subset

# add transforms and download datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# set dataloaders and use DBS training
trainloader = dbstraining.load_merged_trainloader(trainset, batch_sizes_train, ratios, print_info=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Example: Access and print information for the selected batch, and display samples
dbstraining.print_batch_info(trainloader, index=127, display_samples=True)
```

### Investigating Batch Samples

<div align="center">
    <img src="https://raw.githubusercontent.com/starkslab/starkslab.github.io/main/dbstraining/static/images/batch32.png" width="250" style="vertical-align: top;" alt="Batch Size 32 Samples" />
    <img src="https://raw.githubusercontent.com/starkslab/starkslab.github.io/main/dbstraining/static/images/batch64.png" width="250" style="vertical-align: top;" alt="Batch Size 64 Samples" />
    <img src="https://raw.githubusercontent.com/starkslab/starkslab.github.io/main/dbstraining/static/images/batch128.png" width="250" style="vertical-align: top;" alt="Batch Size 128 Samples" />
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