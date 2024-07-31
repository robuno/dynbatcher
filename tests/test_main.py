from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import dbstraining

# Define batch sizes and ratios
ratios = [0.5, 0.3, 0.2]              # Corresponding ratios for each subset
batch_sizes_train = [32, 64, 128]     # Corresponding batch sizes for each subset
batch_size_test = 64

# add transforms and download datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# set dataloaders
trainloader = dbstraining.load_merged_trainloader(trainset, batch_sizes_train, ratios, print_info=True)
testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

# Print the number of batches in the train and test loaders
num_train_batches = len(trainloader)
num_test_batches = len(testloader) 
print("Train batches:", num_train_batches)
print("Test batches:", num_test_batches)

# # Example: Access and print information for the selected batch, and display samples
dbstraining.print_batch_info(trainloader, 127, display_samples=True)
dbstraining.print_batch_info(trainloader, 1200, display_samples=True)
dbstraining.print_batch_info(trainloader, 1311, display_samples=True)