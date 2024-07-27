from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from assignments.knn_classifier import KNNClassifier


def cross_validate_knn(classifier: KNNClassifier, X: np.ndarray, y: np.ndarray,
                       k_choices: np.ndarray, num_folds: int) -> Dict[str, Dict[int, np.ndarray]]:
    """Performs cross-validation for kNN classifier. This is done by following the steps below:
        1. Randomly shuffle the data and labels.
        2. Split the data into k_folds.
        3. For each k in k_choices:
            3.1. For each fold:
                3.1.1. Train the classifier on the training data.
                3.1.2. Predict labels for the validation data.
            3.2. Compute the metrics for the current k.
        4. Return the metrics for all k.

    Args:
        classifier (KNNClassifier): The kNN classifier to be cross-validated.
        X (numpy.ndarray): The training data features of shape (N, D), where N is the number of data points and D is
            the number of features.
        y (numpy.ndarray): The training data labels of shape (N,).
        k_choices (numpy.ndarray): An array of k values.
        num_folds (int): The number of folds to be used for cross-validation.

    Returns:
        dict: A dictionary containing the results of cross-validation. The keys are the names of the metrics and the
            values are dictionaries containing the metric values for different values of k. The resulting values should
            be mean values over all folds.

            Metrics:
                - accuracy: The accuracy of the classifier computed by `sklearn.metrics.accuracy_score()`.
                    The value for each k is a scalar.
                - precision: The precision of the classifier computed by `sklearn.metrics.precision_score()`.
                    The value for each k is a numpy array of shape (num_classes).
                - recall: The recall of the classifier computed by `sklearn.metrics.recall_score()`.
                    The value for each k is a numpy array of shape (num_classes).
                - f1: The f1 score of the classifier computed by `sklearn.metrics.f1_score()`.
                    The value for each k is a numpy array of shape (num_classes).

            Important: The precision, recall, and f1 metrics are computed for each class separately, you should use
                `sklearn.metrics.precision_score(..., average=None)` to compute the metrics for each class.

            Example: If we have 3 classes and 4 folds and k_choices = [1, 3, 5, 7], then the dictionary returned
                by this function will be:

            k_to_metrics = {
                'accuracy': {
                    1: 0.9,
                    3: 0.1,
                    5: 0.5,
                    7: 0.1
                },
                'precision': {
                    1: [0.9, 0.8, 0.7],
                    3: [0.1, 0.3, 0.6],
                    5: [0.5, 0.2, 0.3],
                    7: [0.1, 0.1, 0.8]
                },
                'recall': {
                    1: [0.6, 0.5, 0.4],
                    ...
                },
                'f1': {
                    1: [0.8, 0.7, 0.6],
                    ...
                }
            }
    """

    k_to_metrics = dict()
    k_to_metrics["accuracy"] = dict()
    k_to_metrics["precision"] = dict()
    k_to_metrics["recall"] = dict()
    k_to_metrics["f1"] = dict()

    # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 2.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
    # TODO:                                                             #
    # Implement cross-validation for kNN classifier. Use ChatGPT to     #
    # understand the k-fold cross-validation procedure. You may use     #
    # ChatGPT to generate the code for this function.                   #
    #                                                                   #
    # HINT: You may find the following functions useful:                #
    #      - np.random.shuffle                                          #
    #      - np.array_split                                             #
    #      - sklearn.metrics.accuracy_score                             #
    #      - sklearn.metrics.precision_score                            #
    #      - sklearn.metrics.recall_score                               #
    #      - sklearn.metrics.f1_score                                   #
    #                                                                   #
    # Good luck!                                                        #
    # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
    # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
    #
    #                    ╔═══════════════════════╗
    #                    ║                       ║
    #                    ║       YOUR CODE       ║
    #                    ║                       ║
    #                    ╚═══════════════════════╝
    #
    # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    return k_to_metrics
