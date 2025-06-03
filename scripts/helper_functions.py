"""
A series of helper functions used throughout the course.
Reusable utilities for data processing, plotting, and model evaluation.
"""

# Standard library
import os
import zipfile
from pathlib import Path
from typing import List

# Third-party libraries
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Args:
        dir_path (str): Target directory path.

    Prints:
        Number of directories and images in each subdirectory.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"{len(dirnames)} dirs & {len(filenames)} imgs in '{dirpath}'.")


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots decision boundaries of a model's predictions on X in comparison to y.

    Adapted from: https://madewithml.com/courses/foundations/neural-networks/

    Args:
        model (torch.nn.Module): Trained classification model.
        X (torch.Tensor): Feature data.
        y (torch.Tensor): Target labels.
    """
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # multi-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=cm.get_cmap("RdYlBu"), alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=cm.get_cmap("RdYlBu"))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
    Plots training and test data and optionally predictions.

    Args:
        train_data: Training feature data.
        train_labels: Training target labels.
        test_data: Test feature data.
        test_labels: Test target labels.
        predictions: (Optional) Model predictions on test data.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Calculates accuracy between true and predicted labels.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        float: Accuracy percentage.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def print_train_time(start: float, end: float, device=None):
    """
    Prints the training time in seconds.

    Args:
        start (float): Start time.
        end (float): End time.
        device (optional): Computing device info.

    Returns:
        float: Training time in seconds.
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def plot_loss_curves(results: dict):
    """
    Plots training and test loss and accuracy curves.

    Args:
        results (dict): Dictionary with train/test loss and accuracy lists.
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Train accuracy")
    plt.plot(epochs, test_accuracy, label="Test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,  # type: ignore
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",  # type: ignore
):
    """
    Predicts and plots a single image with model's prediction.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        image_path (str): Path to target image.
        class_names (List[str], optional): Class names for predictions.
        transform (callable, optional): Image transform.
        device (torch.device): Device to use (CPU/GPU).
    """
    target_image = (
        torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.0
    )

    if transform:
        target_image = transform(target_image)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.imshow(target_image.squeeze().permute(1, 2, 0))
    title = (
        f"Pred: {class_names[target_image_pred_label.cpu()]} | "
        f"Prob: {target_image_pred_probs.max().cpu():.3f}"
        if class_names
        else f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    )
    plt.title(title)
    plt.axis(False)


def set_seeds(seed: int = 42):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """
    Downloads and unzips a dataset from a source URL.

    Args:
        source (str): URL to the zipped data.
        destination (str): Target folder name.
        remove_source (bool): If True, removes zip file after extraction.

    Returns:
        Path: Path to the unzipped data folder.
    """
    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        if remove_source:
            os.remove(data_path / target_file)

    return image_path
