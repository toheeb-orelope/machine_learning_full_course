# Import necessary libraries
import numpy as np
from collections import Counter

# KNN (K Nearest Neighbors) implementation


# Function to calculate the Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# KNN classifier class
class KNN:
    # Initialize the classifier with the number of neighbors k
    def __init__(self, k=3):
        self.k = k

    # Store the training data and labels
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Predict the labels for a set of input samples
    def predict(self, X):
        # For each sample in X, predict its label
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # Predict the label for a single sample
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of the k nearest samples
        k_indices = np.argsort(distances)[: self.k]
        # Get the labels of the k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote: return the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
