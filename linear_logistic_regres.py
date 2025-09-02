import numpy as np


class BaseRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def approximation(self, X, w, b):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _prediction(self, X, w, b):
        raise NotImplementedError("This method should be overridden by subclasses")


class LinearRegression(BaseRegression):

    def approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _prediction(self, X, w, b):
        return np.dot(X, w) + b


class LogisticRegression(BaseRegression):

    def approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def _prediction(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
