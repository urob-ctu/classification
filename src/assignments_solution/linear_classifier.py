import torch
import numpy as np
import torch.nn as nn

class LinearClassifier:
    def __init__(self, num_features: int, 
                 num_classes: int, 
                 learning_rate: float = 1e-3,
                 batch_size: int = 100, 
                 weight_scale: float = 1e-3,
                 reg: float = 1e-3, 
                 num_iters: int = 1000, 
                 verbose: bool = True):

        self.num_classes = num_classes
        self.num_features = num_features

        self.reg = reg
        self.verbose = verbose
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize the weights (Xavier init) and biases (zero init)
        self.W = nn.Parameter(torch.randn(num_features, num_classes, dtype=torch.float) 
                              * np.sqrt(2 / (num_features + num_classes))) 
        self.W.data = self.W.data * weight_scale  # Scale the weights
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float))

    def load_weights(self, W: torch.Tensor, b: torch.Tensor) -> None:
        """ Load the weights and biases into the model.

        Args:
            W: The weights of shape (D, C)
            b: The biases of shape (C,)

        Returns:
            None
        """

        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """ Predict the labels of the data.

        Args:
            X: Input data of shape (N, D)

        Returns:
            y_pred: The predicted labels of the data. Array of shape (N,)
        """

        logits = self.forward(X)
        return torch.argmax(logits, axis=1)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> list:
        loss_history = []
        for i in range(self.num_iters):
            batch_indices = torch.randint(0, X.shape[0], (self.batch_size,))
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            logits = self.forward(X_batch)
            loss = torch.tensor([0.0], requires_grad=True)
            self._zero_gradients()

            # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
            # TODO:                                                             #
            # Implement one iteration of the training loop. Use the computed    #
            # scores to compute the Cross Entropy Loss, add the regularization  #
            # loss of all parameters and store it to the variable `loss`.       #
            # Then, compute the backward pass and update the weights and biases #
            # of the model. After that zero out the gradients of the weights    #
            # and biases.                                                       #
            #                                                                   #
            # HINT: - Use only already implemented functions of the Tensor      #
            #         class.                                                    #
            #       - Do not forget to add the regularization loss              #
            #         (defined in Tensor class) of ALL parameters. Use self.reg #
            #         as the regularization strength.                           #
            #       - Call step() on the `loss` variable to update              #
            #         the parameters.                                           #
            #                                                                   #
            # Good luck!                                                        #
            # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
            # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)


            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, y_batch)
            
            loss = loss + self.reg * torch.sum(self.W ** 2)
            loss = loss + self.reg * torch.sum(self.b ** 2)

            loss.backward(retain_graph=True)
            
            with torch.no_grad():
                self.W.data = self.W.data - self.learning_rate * self.W.grad
                self.b.data = self.b.data - self.learning_rate * self.b.grad

            # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

            loss_history.append(loss.data)
            if self.verbose and i % 100 == 0:
                print(f"iteration {i} / {self.num_iters}: {loss.data}")

        return loss_history

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Compute the logits of the model.

        Args:
            X: Input data of shape (N, D)

        Returns:
            logits: The logits of the model. Tensor of shape (N, C)

        """
        logits = torch.zeros((X.shape[0], self.num_classes))

        # # If X is a numpy array, convert it to a tensor
        # if isinstance(X, np.ndarray):
        #     X = torch.from_numpy(X).float()

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement computation of the logits of the model.                 #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

        logits = X @ self.W + self.b

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return logits
    
    def _zero_gradients(self):
        if self.W.grad is not None:
            self.W.grad.zero_()
        if self.b.grad is not None:
            self.b.grad.zero_()
