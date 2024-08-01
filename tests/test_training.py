import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dynbatcher.utils import plot_results
from tqdm import tqdm
import dynbatcher

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main ():
    # Define batch sizes and ratios
    ratios = [0.5, 0.3, 0.2]              # Corresponding ratios for each subset
    batch_sizes_train = [32, 64, 128]     # Corresponding batch sizes for each subset
    batch_size_test = 64

    # add transforms and download datasets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # set dataloaders
    trainloader = dynbatcher.load_merged_trainloader(trainset, batch_sizes_train, ratios)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    # Print the number of batches in the train and test loaders
    num_train_batches = len(trainloader)
    num_test_batches = len(testloader) 
    print("Train batches:", num_train_batches)
    print("Test batches:", num_test_batches)

    ### Training the NN
    # Define the model, criterion, and optimizer
    NUM_EPOCHS = 2
    LR = 0.001
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader, desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}'), 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_losses.append(running_loss / (i + 1))
            train_accuracies.append(100 * correct / total)

        epoch_loss = running_loss / num_train_batches
        epoch_acc = 100 * correct / total
        tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        for j, data in enumerate(tqdm(testloader, desc=f'Testing Epoch {epoch+1}/{NUM_EPOCHS}'), 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_losses.append(running_loss / (j + 1))
            test_accuracies.append(100 * correct / total)

        epoch_loss = running_loss / num_test_batches
        epoch_acc = 100 * correct / total
        tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS} - Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.2f}%")

    # Plot results and save
    plot_results(
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        num_batches_train=num_train_batches,
        num_batches_test=num_test_batches,
        epochs=NUM_EPOCHS
    )


if __name__ == "__main__":
    main()