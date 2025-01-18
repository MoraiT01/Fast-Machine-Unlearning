"""Files for metrics to evaluate models and MU algorithms."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Dict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of classes
num_classes = 10

# Initialize tensors to track correct predictions and total samples per class
correct_counts = None # torch.zeros(num_classes, dtype=torch.float32)
cumulative_loss = None # torch.zeros(num_classes, dtype=torch.float32)
total_counts = None # torch.zeros(num_classes, dtype=torch.float32)

def update_class_accuracy(predictions, labels):
    """
    Update the accuracy counts for each class.
    Args:
        predictions (Tensor): Tensor of predicted class indices (batch_size,).
        labels (Tensor): Tensor of true class labels (batch_size,).
    """
    global correct_counts, total_counts
    for cls in range(num_classes):
        cls_mask = (labels == cls)  # Mask for samples of the current class
        correct_counts[cls] += torch.sum((predictions == labels) & cls_mask).item()
        total_counts[cls] += torch.sum(cls_mask).item()

def update_class_loss(losses, labels):
    """
    Update the cumulative loss for each class.
    Args:
        losses (Tensor): Tensor of individual sample losses (batch_size,).
        labels (Tensor): Tensor of true class labels (batch_size,).
    """
    global cumulative_loss, total_counts
    for cls in range(num_classes):
        cls_mask = (labels == cls)  # Mask for samples of the current class
        cumulative_loss[cls] += torch.sum(losses[cls_mask]).item()
        total_counts[cls] += torch.sum(cls_mask).item()

def get_cumulative_accuracy():
    """
    Calculate the cumulative average accuracy or loss for each class.
    Returns:
        Tensor: Tensor of cumulative average accuracies or losses for each class.
    """
    return correct_counts / (total_counts + 1e-10)  # Avoid division by zero

def get_cumulative_average_loss():
    """
    Calculate the cumulative average loss for each class.
    Returns:
        Tensor: Tensor of cumulative average losses for each class.
    """
    return cumulative_loss / (total_counts + 1e-10)  # Avoid division by zero


def calc_accuracy(model: torch.nn.Module, testing_loader: DataLoader,) -> float:
    """
    Calculates the accuracy of the model on parsed data

    Args:
        model (Module): The model to be evaluated
        testing_loader (DataLoader): A DataLoader containing the evaluation data
        n (int): The number of the model to be evaluated
        total (int): The total number of models to be evaluated

    Returns:
        float: The average accuracy of the model on the evaluation data
    """
    
    # evaluation phase
    model.eval()  # Set model to evaluation mode

    global correct_counts, total_counts, num_classes
    correct_counts = torch.zeros(num_classes, dtype=torch.float32)
    total_counts = torch.zeros(num_classes, dtype=torch.float32)

    # Iterate over the evaluation dataset
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in tqdm(testing_loader, desc=f"Evaluation model", unit="batch", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            _, l = torch.max(labels, 1)
            update_class_accuracy(preds, l)

    return get_cumulative_accuracy()

def calc_loss(model: torch.nn.Module, testing_loader: DataLoader, loss_function = nn.CrossEntropyLoss()) -> float:
    """
    Calculates the accuracy of the model on parsed data

    Args:
        model (Module): The model to be evaluated
        testing_loader (DataLoader): A DataLoader containing the evaluation data
        loss_function (Module): The loss function to be used for evaluation

    Returns:
        float: The average loss of the model on the evaluation data
    """

    # evaluation phase
    model.eval()  # Set model to evaluation mode

    global correct_counts, total_counts, num_classes
    correct_counts = torch.zeros(num_classes, dtype=torch.float32)
    total_counts = torch.zeros(num_classes, dtype=torch.float32)

    # Iterate over the evaluation dataset
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in tqdm(testing_loader, desc=f"Evaluation model", unit="batch", leave=False):
            # Move the images and labels to the device (CPU or GPU)
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = loss_function(outputs, labels)
            # Append the loss to the list of losses
            update_class_loss(loss, labels)

    return get_cumulative_average_loss()

def kl_divergence_between_models(model1: torch.nn.Module, model2: torch.nn.Module, data_loader: DataLoader, device='cpu') -> float:
    """
    Compute the average KL divergence between prediction distributions of two models.
    
    Args:
        model1 (torch.nn.Module): The first model.
        model2 (torch.nn.Module): The second model.
        data_loader (DataLoader): A DataLoader for the dataset to be used for comparison.
        device (str): Device to run the models on ('cpu' or 'cuda').

    Returns:
        float: The average KL divergence between the two models' prediction distributions.
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    
    # Cumulative Average KL Divergence
    kl_divergence_ca = 0
    n = 0
    very_small_number = 1e-6
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"KL Divergence", unit="batch", leave=False): # Iterate over data_loader:
            n += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            # Get prediction logits from both models
            probs1 = model1(inputs)
            probs2 = model2(inputs)

            # Add a small number to avoid log(0) errors
            probs1 = probs1 + very_small_number
            probs2 = probs2 + very_small_number

            # Ensure probs are probabilities (apply softmax if needed)
            probs1 = F.softmax(probs1, dim=1)
            probs2 = F.softmax(probs2, dim=1)

            # Calculate the KL divergence for each sample and sum up
            kl_divergence = F.kl_div(probs1.log(), probs2, reduction='batchmean').item()

            # kl_divergence = round(kl_divergence, 4) # I had problems with very small numbers accumulating to scuw the result
            
            # Update cumulative average
            kl_divergence_ca = kl_divergence_ca + (kl_divergence - kl_divergence_ca)/n
    
    # Return average KL divergence over all samples
    return kl_divergence_ca

def calc_mutlimodel_metric_average(modeltype1: Dict[str, torch.nn.Module], modeltype2: Dict[str, torch.nn.Module], testing_loader: DataLoader = None,) -> float:
    
    result = 0.0
    counter = 0
    if len(modeltype1.keys()) != len(modeltype2.keys()):
        raise ValueError("modeltype1 and modeltype2 must have the same keys and same length")
    for idx in modeltype1.keys():
        result += kl_divergence_between_models(modeltype1[idx], modeltype2[idx], testing_loader,)
        counter += 1

    return result / counter