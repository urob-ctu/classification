from typing import List, Tuple

import numpy as np


def reshape_to_vectors(*arrays: np.ndarray) -> List[np.ndarray]:
    """Reshape the input arrays to 2D vectors, preserving the first dimension.

    Args:
        *arrays (np.ndarray): One or more NumPy arrays to be reshaped.

    Returns:
        List[np.ndarray]: A list of reshaped 2D arrays, where each array is 
        flattened except for the first dimension.
    """
    return [array.reshape(array.shape[0], -1) for array in arrays]


def normalize(*arrays: np.ndarray) -> List[np.ndarray]:
    """Normalize the input arrays using the mean and standard deviation.

    Args:
        *arrays (np.ndarray): One or more NumPy arrays to be normalized.

    Returns:
        List[np.ndarray]: A list of normalized arrays with mean 0 and standard deviation 1.

    Note:
        All arrays are normalized using a common mean and standard deviation computed 
        from concatenating the input arrays along the first axis.
    """
    # Calculate the common mean and standard deviation
    mean = np.mean(np.concatenate(arrays), axis=0)
    std = np.std(np.concatenate(arrays), axis=0)
    return [(array - mean) / std for array in arrays]


def dataset_stats(X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray, 
                  verbose: bool = False) -> Tuple[int, int, int]:
    """Compute and optionally print statistics of the dataset.

    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels.
        verbose (bool, optional): If True, prints detailed statistics. Default is False.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - The number of features per sample.
            - The number of unique classes.
            - The total number of samples across training, validation, and test datasets.
    """
    num_features = X_train[0].size
    num_classes = len(np.unique(y_train))
    num_samples = len(X_train) + len(X_val) + len(X_test)

    if verbose:
        # Print the shapes of the data
        print('---------------- Training data ----------------')
        print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')

        print('\n---------------- Validation data ----------------')
        print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

        print('\n---------------- Testing data ----------------')
        print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

        print('\n---------------- Dataset info ----------------')
        print(f'Number of classes: {num_classes}')
        print(f'Number of features: {num_features}')
        print(f'Number of samples in dataset: {num_samples}')
        print(f'Number of samples in training set: {len(X_train)}, '
              f'which is {100 * len(X_train) / num_samples:.2f}% of the dataset')
        print(f'Number of samples in validation set: {len(X_val)}, '
              f'which is {100 * len(X_val) / num_samples:.2f}% of the dataset')
        print(f'Number of samples in testing set: {len(X_test)}, '
              f'which is {100 * len(X_test) / num_samples:.2f}% of the dataset')
    
    return num_features, num_classes, num_samples


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """Calculate the exponential moving average of a data sequence.

    Args:
        data (np.ndarray): The input data sequence.
        alpha (float): The smoothing factor for the EMA.

    Returns:
        np.ndarray: The exponential moving average of the input data.
    """
    ema = np.zeros(data.size)
    ema[0] = data[0]
    for i in range(1, data.size):
        weights = np.flip((1 - alpha) ** np.arange(i + 1))
        weighted_sum = np.sum(weights * data[:i + 1])
        ema[i] = weighted_sum / np.sum(weights)
    return ema


def dynamic_ema(data: np.ndarray) -> np.ndarray:
    """Calculate the exponential moving average with a dynamically adjusted alpha.

    Args:
        data (np.ndarray): The input data sequence.

    Returns:
        np.ndarray: The exponential moving average of the input data using a dynamic alpha.

    Note:
        The alpha value is dynamically calculated based on the standard deviation of the 
        data over a moving window, enhancing the EMA's responsiveness to changes in data variance.
    """
    mean_std = mean_standard_deviation(data, 100)
    alpha = 2 / (mean_std + 1)
    print(f"alpha = {alpha}")
    return exponential_moving_average(data, alpha)


def mean_standard_deviation(data: np.ndarray, window: int) -> float:
    """Compute the mean of the standard deviations over a moving window.

    Args:
        data (np.ndarray): The input data sequence.
        window (int): The size of the moving window.

    Returns:
        float: The maximum standard deviation over the moving window.
    
    Note:
        This function is used to dynamically adjust the smoothing factor for the EMA calculation.
    """
    stds = np.zeros(data.size)
    for i in range(data.size):
        stds[i] = np.std(data[max(0, i - window):i + 1])
    return np.max(stds)



