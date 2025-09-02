import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        # Learning rate controls the step size in gradient descent
        self.lr = lr
        # Number of iterations for training
        self.n_iters = n_iters
        # Model parameters (weights and bias) initialized later
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Get number of samples and features from input data
        n_samples, n_features = X.shape
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent algorithm
        for _ in range(self.n_iters):
            # Calculate predicted values using current weights and bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients for weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias using gradients
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Predict output values using learned weights and bias
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
