import matplotlib.pyplot as plt
import datetime
import os

def check_and_create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def print_batch_info(train_dataloader, index, display_samples=False):
    """
    Print batch information and optionally display samples.

    Args:
        train_dataloader (DataLoader): The DataLoader with training batches.
        index (int): The batch index.
        display_samples (bool, optional): If True, display samples in a grid. Default is False.

    This function:
    1. Ensures the 'plots/batches' directory exists.
    2. Retrieves the specified batch and prints the number of samples.
    3. If display_samples is True, displays the images and labels in a grid and saves the plot.
    """

    DIR_BATCHES = "plots/batches"
    check_and_create_dir(DIR_BATCHES)

    batch = train_dataloader.get_batch_by_index(index)
    inputs, labels = batch
    num_samples = inputs.shape[0]
    print(f"Batch {index} - Number of samples: {num_samples}")

    if display_samples:
        # Determine the grid size
        num_rows = (num_samples + 9) // 10  # ceil division to get the number of rows
        fig, axes = plt.subplots(num_rows, 10, figsize=(10, 1.5 * num_rows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, (img, label) in enumerate(zip(inputs, labels)):
            img = img.squeeze()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"#{i+1}: {label.item()}")
            axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

        timestamp = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        fig.savefig(f'{DIR_BATCHES}/batch_{index}_({num_samples})_{timestamp}.png')
            

def plot_results(train_losses, test_losses, train_accuracies, test_accuracies, num_batches_train, num_batches_test, epochs):
    check_and_create_dir('plots')
        
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
    timestamp = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
    fig.savefig(f'plots/results_{timestamp}.png')
