import torch

def get_time_diff(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def binary_accuracy(predictions, labels):
    rounded_preds = torch.round(torch.sigmoid(predictions))
    correct = (rounded_preds == labels).float()
    acc = correct.sum() / len(correct)
    return acc

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

import matplotlib.pyplot as plt

def plot_losses(train_loss, valid_loss, output):
    plt.plot(train_loss, 'r', label="Training Loss")
    plt.plot(valid_loss, 'g', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(output)