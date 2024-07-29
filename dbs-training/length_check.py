def print_lengths_and_check(train_losses, test_losses, train_accuracies, test_accuracies, num_train_batches, num_test_batches, epochs):
    expected_train_losses_len = num_train_batches * epochs
    expected_test_losses_len = num_test_batches * epochs

    print(f"Expected train losses length: {expected_train_losses_len}, Actual: {len(train_losses)}")
    print(f"Expected test losses length: {expected_test_losses_len}, Actual: {len(test_losses)}")
    print(f"Expected train accuracies length: {expected_train_losses_len}, Actual: {len(train_accuracies)}")
    print(f"Expected test accuracies length: {expected_test_losses_len}, Actual: {len(test_accuracies)}")

    if len(train_losses) == expected_train_losses_len:
        print("Train losses length is correct.")
    else:
        print("Train losses length is incorrect.")

    if len(test_losses) == expected_test_losses_len:
        print("Test losses length is correct.")
    else:
        print("Test losses length is incorrect.")

    if len(train_accuracies) == expected_train_losses_len:
        print("Train accuracies length is correct.")
    else:
        print("Train accuracies length is incorrect.")

    if len(test_accuracies) == expected_test_losses_len:
        print("Test accuracies length is correct.")
    else:
        print("Test accuracies length is incorrect.")
