import os
import pickle
import subprocess
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import importlib.util
from typing import List, Tuple, Union

import yaml
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt

from .plot import visualize_cifar10

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