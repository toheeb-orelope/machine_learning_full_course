import numpy as np


# Perceptron classifier implementation
class Perceptron:

    def __init__(self, lr=0.01, n_iter=1000):
        # Learning rate and number of iterations
        self.lr = lr
        self.n_iter = n_iter
        # Activation function (unit step)
        self.activation_func = self._unit_step_func
        # Model parameters (weights and bias)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Get number of samples and features
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to 0 and 1
        y_ = np.array([1 if i > 0 else 0 for i in y])
        # Training loop
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                # Compute linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply activation function
                y_predicted = self.activation_func(linear_output)

                # Update weights and bias
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        # Predict class labels for samples in X
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, X):
        # Unit step activation function
        return np.where(X >= 0, 1, 0)
