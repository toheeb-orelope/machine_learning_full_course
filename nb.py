import numpy as np


class NaiveBayes:
    def __init__(self, X, y):
        # Get number of samples and features
        n_samples, n_features = X.shape
        # Find unique class labels
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, variance, and prior probability arrays
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and priors for each class
        for c in self._classes:
            X_c = X[y == c]  # Select samples of class c
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        # Predict class for each sample in X
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # Compute posterior probability for each class
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # Log prior
            class_conditional = np.sum(np.log(self._pdf(idx, x)))  # Log likelihood
            posterior = prior + class_conditional
            posteriors.append(posterior)
        # Return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Calculate Gaussian probability density function for each feature
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
