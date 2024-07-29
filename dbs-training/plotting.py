import matplotlib.pyplot as plt
import datetime

def get_epoch_vals(batch_values, num_batches_per_epoch):
    epoch_vals = []
    num_epochs = len(batch_values) // num_batches_per_epoch
    for ep in range(1, num_epochs + 1):
        epoch_vals.append(batch_values[(num_batches_per_epoch * ep) - 1])
    return epoch_vals

def plot_results(train_losses, test_losses, train_accuracies, test_accuracies, num_batches_train, num_batches_test, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    for epoch in range(1, epochs):
        ax1.axvline(x=epoch * num_batches_train, color='gray', linestyle='--', alpha=0.2)

    for epoch in range(1, epochs):
        ax1.axvline(x=epoch * num_batches_test, color='green', linestyle='dotted', alpha=0.2)
    ax1.set_title('Losses')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    for epoch in range(1, epochs):
        ax2.axvline(x=epoch * num_batches_train, color='gray', linestyle='--', alpha=0.2)
    for epoch in range(1, epochs):
        ax2.axvline(x=epoch * num_batches_test, color='green', linestyle='dotted', alpha=0.2)
    ax2.set_title('Accuracies')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.show()
    # Save the figure with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'plots/results_{timestamp}.png')

def plot_epoch_results(epoch_train_losses, epoch_test_losses, epoch_train_accuracies, epoch_test_accuracies, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(range(1, epochs + 1), epoch_train_losses, label='Train Loss', marker='o')
    ax1.plot(range(1, epochs + 1), epoch_test_losses, label='Test Loss', marker='o')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(range(1, epochs + 1), epoch_train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(range(1, epochs + 1), epoch_test_accuracies, label='Test Accuracy', marker='o')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.show()
    # Save the figure with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'plots/epoch_results_{timestamp}.png')
