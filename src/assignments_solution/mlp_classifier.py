from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class MLPClassifier:
    def __init__(self, input_size: int, 
                 hidden_dim_1: int, 
                 hidden_dim_2: int,
                 output_size: int, 
                 activation: str = 'relu',
                 learning_rate: float = 1e-3, 
                 reg: float = 1e-6, 
                 batch_size: int = 100,
                 num_iters: int = 1000, 
                 verbose: bool = False):

        self.params = dict(
            W1=nn.Parameter(torch.randn(input_size, hidden_dim_1, dtype=torch.float) * 1),
            b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
            W2=nn.Parameter(torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float) * 1),
            b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
            W3=nn.Parameter(torch.randn(hidden_dim_2, output_size, dtype=torch.float) * 1),
            b3=nn.Parameter(torch.zeros(output_size, dtype=torch.float))
        )

        self.reg = reg
        self.verbose = verbose
        self.num_iters = num_iters
        self.activation = activation
        self.batch_size = batch_size
        self.num_classes = output_size
        self.learning_rate = learning_rate

        self.activation_func = None

        if activation == 'relu':
            self.activation_func = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation_func = nn.Tanh()


    def train(self, 
              X_train: torch.Tensor, 
              y_train: torch.Tensor, 
              X_val: torch.Tensor, 
              y_val: torch.Tensor) -> tuple:
        
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
            if i % 500 == 0 or i == self.num_iters - 1:

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            X (Union[np.ndarray, Tensor]): Input data of shape (N, D)

        Returns:
            Tensor: Output data of shape (N, C)
        """

        out = self._first_layer(X)
        out = self._second_layer(out)
        logits = self._output_layer(out)

        return logits

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict the class labels for the provided data.

        Args:
            X (np.ndarray): Input data of shape (N, D)
            zero_grad (bool, optional): Whether to zero the gradients after
                prediction. Defaults to False.

        Returns:
            np.ndarray: Predicted class labels of shape (N,)
        """

        logits = self.forward(X)
        y_pred = torch.argmax(logits, axis=1)

        return y_pred

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the neural network.

        Args:
            X (np.ndarray): Input data of shape (N, D)
            y (np.ndarray): Labels of shape (N,)
            zero_grad (bool, optional): Whether to zero the gradients after
                computing the loss. Defaults to False.

        Returns:
            torch.Tensor: Loss of the neural network
        """

        loss_fn = torch.nn.CrossEntropyLoss()
        
        logits = self.forward(X)
        loss = loss_fn(logits, y)

        for name in self.params.keys():
            loss = loss + self.reg * torch.sum(self.params[name] ** 2)

        return loss

    def _first_layer(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the first layer of the MLP.

        Args:
            X: Input data of shape (N, D)

        Returns:
            out: Output data of shape (N, H1)
        """

        out = None

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 5.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

        out = X @ self.params['W1'] + self.params['b1']
        out = self.activation_func(out)

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return out

    def _second_layer(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the second layer of the MLP.

        Args:
            X: Input data of shape (N, H1)

        Returns:
            out: Output data of shape (N, H2)

        """

        out = None

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 5.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

        out = X @ self.params['W2'] + self.params['b2']
        out = self.activation_func(out)

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return out

    def _output_layer(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the third layer of the MLP.

        Args:
            X: Input data of shape (N, H2)

        Returns:
            out: Output data of shape (N, C)

        """

        out = None

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 5.3 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

        out = X @ self.params['W3'] + self.params['b3']

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return out
    
    def _zero_gradients(self):
        for name in self.params.keys():
            if self.params[name].grad is not None:
                self.params[name].grad.zero_()

    def _update_weights(self):
        with torch.no_grad():
            for name in self.params.keys():
                self.params[name].data = self.params[name].data - self.learning_rate * self.params[name].grad

    # ================ Methods for the animation =================

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = Tensor(X)
        out = self._first_layer(X)
        out = self._second_layer(out)
        return out.data

    def transform_point(self, x: np.ndarray) -> np.ndarray:
        x = x[:2][np.newaxis, :]
        out = self.transform(x)
        return np.array([out[0][0], out[0][1], 0])

    def predict_transformed(self, X: np.ndarray) -> np.ndarray:
        X = Tensor(X)
        scores = self._output_layer(X)
        return np.argmax(scores.data, axis=1)
