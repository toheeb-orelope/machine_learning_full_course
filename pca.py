import numpy as np


# Principal Component Analysis (PCA) implementation
class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        # self.explained_variance = None

    def fit(self, X):
        #mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        #Covariance matrix
        cov = np.cov(X.T)

        #Eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        #Sort eigenvalues and corresponding eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        #Store first n_components eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        #Project data
        X = X - self.mean
        return np.dot(X, self.components.T)
