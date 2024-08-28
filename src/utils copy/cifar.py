import os
import pickle
import subprocess
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def download_cifar10(directory: str) -> None:
    """Downloads and extracts the CIFAR-10 dataset into the specified directory.

    This function checks if the CIFAR-10 dataset is already present in the specified directory.
    If not, it downloads the dataset, extracts it, and moves the extracted files to the directory.

    Args:
        directory (str): The directory where the dataset will be downloaded and extracted.
    
    Returns:
        None
    
    Note:
        Requires `wget` and `tar` to be available in the system's PATH.
    """

    if os.path.exists(directory):
        print("CIFAR-10 dataset already exists")
        return

    # Download CIFAR-10 dataset
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar = "cifar-10-python.tar.gz"
    subprocess.call(["wget", url])

    # Extract and move to the specified directory
    subprocess.call(["tar", "-xvzf", tar])
    subprocess.call(["mv", "cifar-10-batches-py", directory])
    subprocess.call(["rm", tar])


def load_cifar10_bach_file(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single batch file from the CIFAR-10 dataset.

    Args:
        file (str): The path to the batch file to be loaded.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - An array containing the image data of shape (10000, 32, 32, 3).
            - An array containing the corresponding labels of shape (10000,).
    """

    with open(file, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10(directory: str, visualize_samples: str = False) -> Tuple:
    """Loads the CIFAR-10 dataset from the specified directory and splits it into training, validation, and test sets.

    Args:
        directory (str): The directory where the CIFAR-10 dataset is located.
        visualize_samples (bool, optional): If True, visualizes a sample of images from the training set. Default is False.
    
    Returns:
        Tuple containing:
            - X_train (numpy.ndarray): An array of training images.
            - y_train (numpy.ndarray): An array of training labels.
            - X_val (numpy.ndarray): An array of validation images.
            - y_val (numpy.ndarray): An array of validation labels.
            - X_test (numpy.ndarray): An array of test images.
            - y_test (numpy.ndarray): An array of test labels.
    """

    # Download the CIFAR-10 dataset if it does not exist
    download_cifar10(directory)

    # Load the training and test data
    X_train, y_train = [], []
    for i in range(1, 6):
        file = os.path.join(directory, f"data_batch_{i}")
        X, y = load_cifar10_bach_file(file)
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar10_bach_file(os.path.join(directory, "test_batch"))

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    if visualize_samples:
        visualize_cifar10(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_cifar10_subset(directory: str, num_train: int, 
                        num_val: int, num_test: int, 
                        visualize_samples: bool = False) -> Tuple:
    """Loads a balanced subset of the CIFAR-10 dataset with specified numbers of training, validation, and test samples.

    Args:
        directory (str): The directory where the CIFAR-10 dataset is located.
        num_train (int): The number of training samples per class.
        num_val (int): The number of validation samples per class.
        num_test (int): The number of test samples per class.
        visualize_samples (bool, optional): If True, visualizes a sample of images from the training set. Default is False.
    
    Returns:
        Tuple containing:
            - X_train (numpy.ndarray): An array of training images.
            - y_train (numpy.ndarray): An array of training labels.
            - X_val (numpy.ndarray): An array of validation images.
            - y_val (numpy.ndarray): An array of validation labels.
            - X_test (numpy.ndarray): An array of test images.
            - y_test (numpy.ndarray): An array of test labels.
    """
    
    # Load the full CIFAR-10 dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(directory)

    # Get the number of classes in the dataset
    num_classes = len(np.unique(y_train))

    # Select a subset of the data and make sure the classes are balanced
    X_train = np.vstack([X_train[y_train == i][:num_train] for i in range(num_classes)])
    y_train = np.hstack([y_train[y_train == i][:num_train] for i in range(num_classes)]).astype(np.int32)

    X_val = np.vstack([X_val[y_val == i][:num_val] for i in range(num_classes)])
    y_val = np.hstack([y_val[y_val == i][:num_val] for i in range(num_classes)]).astype(np.int32)

    X_test = np.vstack([X_test[y_test == i][:num_test] for i in range(num_classes)])
    y_test = np.hstack([y_test[y_test == i][:num_test] for i in range(num_classes)]).astype(np.int32)

    if visualize_samples:
        visualize_cifar10(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def visualize_cifar10(X: np.ndarray, y: np.ndarray) -> None:
    """Displays a grid of random samples from the CIFAR-10 dataset.

    Args:
        X (numpy.ndarray): An array of images to be visualized.
        y (numpy.ndarray): An array of corresponding labels.
    
    Returns:
        None
    
    Note:
        The function displays a grid of 7 samples for each of the 10 classes in the dataset.
        Each image is shown without axis markings, and the class name is displayed above the first image of each row.
    """

    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

    num_classes = len(classes)
    samples_per_class = 7

    plt.figure(figsize=(num_classes, samples_per_class))

    for label, class_name in enumerate(classes):

        # Find indices of images for the current class.
        indices = np.flatnonzero(y == label)
        indices = np.random.choice(indices, samples_per_class, replace=False)

        for i, idx in enumerate(indices):
            plt_idx = i * num_classes + label + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(class_name)
    plt.show()

