import numpy as np


# Support Vector Machine (SVM) implementation using gradient descent
class SVM:

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        # Learning rate, regularization parameter, and number of iterations
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        # Model parameters (weights and bias)
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent optimization
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if sample is correctly classified with margin
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Regularization and hinge loss gradient
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Predict class labels for samples in X
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
