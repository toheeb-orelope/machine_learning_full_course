import numpy as np
from decision_tree import DecisionTree
from collections import Counter


# Helper function to find the most common label in an array
def most_common_label(y):
    # Return the most common class label in y
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


# Helper function to create a bootstrap sample from the data
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


# Random Forest classifier implementation
class RandomForest:

    def __init__(
        self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None
    ):
        # Number of trees in the forest
        self.n_trees = n_trees
        # Minimum samples required to split a node
        self.min_samples_split = min_samples_split
        # Maximum depth of each tree
        self.max_depth = max_depth
        # Number of features to consider when looking for the best split
        self.n_features = n_features
        # List to store all the trees
        self.trees = []

    def fit(self, X, y):
        # Train each tree on a bootstrap sample of the data
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
            )
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from all trees and select the most common label
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
