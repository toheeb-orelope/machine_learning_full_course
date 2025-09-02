import numpy as np


# Linear Discriminant Analysis (LDA) implementation
class LDA:

    def __init__(self, n_components=None):
        # Number of linear discriminants to keep
        self.n_components = n_components
        self.linear_discriminants_ = None

    def fit(self, X, y):
        n_features = X.shape[1]  # Number of features
        class_labels = np.unique(y)

        # Compute overall mean
        mean_overall = np.mean(X, axis=0)
        # Initialize within-class and between-class scatter matrices
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # Within-class scatter
            S_W += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            # Between-class scatter
            S_B += n_c * (mean_diff).dot(mean_diff.T)

        # Solve the generalized eigenvalue problem for discriminants
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        sorted_indices = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[sorted_indices]
        # Store first n_components linear discriminants
        self.linear_discriminants_ = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project data onto linear discriminants
        return np.dot(X, self.linear_discriminants_.T)
