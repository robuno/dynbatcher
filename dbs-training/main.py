import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import functions from your local files
from data_processing import load_mnist
from length_check import print_lengths_and_check
from plotting import plot_results, get_epoch_vals, plot_epoch_results
from model_utils import SimpleNN  # Assuming SimpleNN is defined here


def main():
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Define batch sizes and ratios
    ratios = [0.5, 0.3, 0.2]
    batch_sizes_train = [32, 64, 128]     # Corresponding batch sizes for each subset
    batch_size_test = 64

    # Load the MNIST data
    trainloader, testloader = load_mnist(
        merge_trainset=True,
        batch_sizes_train=batch_sizes_train,
        batch_size_test=batch_size_test,
        ratios=ratios
    )
    num_train_batches = len(trainloader)
    num_test_batches = len(testloader)    

    # Print the number of batches in the train and test loaders
    print("Train batches:", num_train_batches)
    print("Test batches:", num_test_batches)

    # Define the model, criterion, and optimizer
    NUM_EPOCHS = 5
    model = SimpleNN()

    # Train and Test the Model
    def train_and_test_model(trainloader, testloader, model, epochs=5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(tqdm(trainloader, desc=f'Training Epoch {epoch+1}/{epochs}'), 0):
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
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            for j, data in enumerate(tqdm(testloader, desc=f'Testing Epoch {epoch+1}/{epochs}'), 0):
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
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.2f}%")

        return train_losses, test_losses, train_accuracies, test_accuracies

    train_losses, test_losses, train_accuracies, test_accuracies = train_and_test_model(
        trainloader,
        testloader,
        model,
        epochs=NUM_EPOCHS
    )

    # Plot the results
    plot_results(
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        num_batches_train=num_train_batches,
        num_batches_test=num_test_batches,
        epochs=NUM_EPOCHS
    )

    pass

if __name__ == "__main__":
    main()