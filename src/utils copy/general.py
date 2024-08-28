import importlib.util
from typing import List, Tuple, Union

import yaml
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt


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


def load_module(src_dir: str, module_name: str) -> object:
    """Dynamically load a Python module from a specified directory.

    Args:
        src_dir (str): The directory containing the module file.
        module_name (str): The name of the module to be loaded (without the .py extension).

    Returns:
        object: The loaded module object.

    Note:
        This function requires that the module file exists in the specified directory.
    """
    module_path = f"{src_dir}/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_config(config_path: str, args: dict) -> dict:
    """Load and render a YAML configuration file with template variables.

    Args:
        config_path (str): The path to the YAML configuration file.
        args (dict): A dictionary of arguments to render the template.

    Returns:
        dict: A dictionary representation of the loaded and rendered configuration.
    
    Note:
        The configuration file can contain Jinja2 template syntax, which will be
        replaced with the values provided in the `args` dictionary.
    """
    with open(config_path) as file:
        template = Template(file.read())
    config_text = template.render(**args)
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    return config


def plot(ax: plt.Axes, X: Union[np.ndarray, List[np.ndarray]], Y: Union[np.ndarray, List[np.ndarray]],
         label_names: list = None, x_label: str = '',
         y_label: str = '', title: str = '') -> None:
    """Plot data on a given Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        X (Union[np.ndarray, List[np.ndarray]]): The x-coordinates of the data.
        Y (Union[np.ndarray, List[np.ndarray]]): The y-coordinates of the data.
        label_names (list, optional): A list of labels for the legend. Default is None.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.
        title (str, optional): The title of the plot. Default is an empty string.
    
    Returns:
        None
    
    Note:
        If the x-coordinates have fewer than 20 elements, markers are shown on the plot.
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    legend = label_names is not None and len(X) > 1

    for i in range(len(X)):
        label = label_names[i] if legend else None
        if X[i].size < 20:
            ax.plot(X[i], Y[i], linewidth=1.5, marker='o', label=label)
        else:
            ax.plot(X[i], Y[i], linewidth=1.5, label=label)
    adjust_plot(ax, X, x_label, y_label, title, legend=legend)


def error_bar(ax: plt.Axes, x: np.ndarray, y: np.ndarray, stds: np.ndarray, x_label: str = '',
              y_label: str = '', title: str = '') -> None:
    """Plot data with error bars on a given Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        x (np.ndarray): The x-coordinates of the data.
        y (np.ndarray): The y-coordinates of the data.
        stds (np.ndarray): The standard deviations for the error bars.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.
        title (str, optional): The title of the plot. Default is an empty string.
    
    Returns:
        None
    """
    ax.errorbar(x, y, yerr=stds, linestyle='dotted', linewidth=1.5, marker='o', capsize=4)
    adjust_plot(ax, x, x_label, y_label, title, legend=False)


def adjust_plot(ax: plt.Axes, X: Union[np.ndarray, List[np.ndarray]], x_label: str, y_label: str, title: str,
                legend: bool) -> None:
    """Adjust plot settings such as labels, title, grid, and legend.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to adjust.
        X (Union[np.ndarray, List[np.ndarray]]): The x-coordinates used to determine ticks and limits.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        legend (bool): Whether to display a legend.
    
    Returns:
        None
    """
    unique_x = list(set(np.concatenate(X))) if isinstance(X, list) else X
    min_k, max_k = np.min(unique_x), np.max(unique_x)

    if legend:
        ax.legend()

    if len(unique_x) < 20:
        ax.set_xticks(unique_x)
        ax.set_xlim(min_k - 1, max_k + 0.35 * (max_k - min_k))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)


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


def plot_nn_training(loss_history: dict, accuracy_history: dict, ema: bool = False, alpha: float = 0.1) -> None:
    """Show the training history of a neural network.

    Args:
        loss_history (dict): A dictionary containing the training and validation loss history.
        accuracy_history (dict): A dictionary containing the training and validation accuracy history.
        ema (bool, optional): Whether to apply exponential moving average to the history. Defaults to False.
        alpha (float, optional): The alpha value for the exponential moving average. Defaults to 0.1.

    Returns:
        None

    Note:
        This function creates a plot with two subplots, displaying the loss and accuracy history.
    """
    train_loss_iters = np.array(list(loss_history["train"].keys()))
    train_losses = np.array(list(loss_history["train"].values()))

    val_loss_iters = np.array(list(loss_history["val"].keys()))
    val_losses = np.array(list(loss_history["val"].values()))

    train_acc_iters = np.array(list(accuracy_history["train"].keys()))
    train_accs = np.array(list(accuracy_history["train"].values()))

    val_acc_iters = np.array(list(accuracy_history["val"].keys()))
    val_accs = np.array(list(accuracy_history["val"].values()))

    if ema:
        train_losses = exponential_moving_average(train_losses, alpha)
        val_losses = exponential_moving_average(val_losses, alpha)
        train_accs = exponential_moving_average(train_accs, alpha)
        val_accs = exponential_moving_average(val_accs, alpha)

    X_acc = [train_acc_iters, val_acc_iters]
    Y_acc = [train_accs, val_accs]

    X_loss = [train_loss_iters, val_loss_iters]
    Y_loss = [train_losses, val_losses]

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot(axes[0], X_loss, Y_loss, ['Training', 'Validation'],
         'Iterations', 'Loss', 'Loss History')
    plot(axes[1], X_acc, Y_acc, ['Training', 'Validation'],
         'Iterations', 'Accuracy', 'Accuracy History')

    plt.show()

