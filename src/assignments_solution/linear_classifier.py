from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class LinearClassifier:
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        weight_scale: float = 1e-3,
        reg: float = 1e-3,
        num_iters: int = 1000,
    ):

        self.num_classes = num_classes
        self.num_features = num_features

        self.reg = reg
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.params = dict(
            W=nn.Parameter(
                torch.randn(num_features, num_classes, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
        )

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple:

        # Initialize the best validation accuracy and the best parameters
        best_val_acc = 0
        best_params = dict()

        # Initialize the loss and accuracy history
        loss_history = dict(train=dict(), val=dict())
        acc_history = dict(train=dict(), val=dict())

        # Training loop
        for i in tqdm(range(self.num_iters), desc="Training"):

            # Select a random batch of data
            batch_indices = torch.randint(0, X_train.shape[0], (self.batch_size,))
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Zero the gradients
            self._zero_gradients()

            # Compute the loss and backpropagate
            train_loss = self.loss(X_batch, y_batch)
            train_loss.backward(retain_graph=True)
            self._update_weights()

            # Save the training loss
            loss_history["train"][i] = train_loss.data

            # Every 500 iterations, compute the validation loss and accuracy
            if i % 100 == 0 or i == self.num_iters - 1:

                # Compute the validation loss
                with torch.no_grad():
                    val_loss = self.loss(X_val, y_val)
                loss_history["val"][i] = val_loss.data

                # Predict the labels for the training and validation data
                y_pred_train = self.predict(X_train)
                y_pred_val = self.predict(X_val)

                # Compute the training and validation accuracy from the predicted labels
                acc_history["train"][i] = accuracy_score(y_train, y_pred_train)
                acc_history["val"][i] = accuracy_score(y_val, y_pred_val)

                # If the current validation accuracy is the best so far, save the parameters
                if acc_history["val"][i] > best_val_acc:
                    best_val_acc = acc_history["val"][i]
                    best_params = deepcopy(self.params)

        # Update the parameters with the best ones
        self.params = best_params

        return loss_history, acc_history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the labels of the data.

        Args:
            X: Input data of shape (N, D)

        Returns:
            y_pred: The predicted labels of the data. Array of shape (N,)
        """

        logits = self.forward(X)
        return torch.argmax(logits, axis=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the logits of the model.

        Args:
            X: Input data of shape (N, D)

        Returns:
            logits: The logits of the model. Tensor of shape (N, C)

        """
        logits = torch.zeros((X.shape[0], self.num_classes))

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement computation of the logits of the model.                 #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

        logits = X @ self.params["W"] + self.params["b"]

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return logits

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor([0.0], requires_grad=True)

        loss_fn = torch.nn.CrossEntropyLoss()

        logits = self.forward(X)
        loss = loss_fn(logits, y)

        for name in self.params.keys():
            loss = loss + self.reg * torch.sum(self.params[name] ** 2)

        return loss

    def _zero_gradients(self):
        for name in self.params.keys():
            if self.params[name].grad is not None:
                self.params[name].grad.zero_()

    def _update_weights(self):
        with torch.no_grad():
            for name in self.params.keys():
                self.params[name].data = (
                    self.params[name].data - self.learning_rate * self.params[name].grad
                )
