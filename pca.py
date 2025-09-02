import numpy as np


# Principal Component Analysis (PCA) implementation
class PCA:

    def __init__(self, n_components):
        # Number of principal components to keep
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Mean centering the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # Compute covariance matrix
        cov = np.cov(X.T)

        # Compute eigenvectors and eigenvalues of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # Store first n_components eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project data onto principal components
        X = X - self.mean
        return np.dot(X, self.components.T)
