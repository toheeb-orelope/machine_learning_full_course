import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nb import NaiveBayes


def accuracy(y_true, y_pred):
    # Calculate the accuracy as the fraction of correct predictions
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Generate a synthetic classification dataset
X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, n_informative=5, random_state=123
)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Initialize and train the Naive Bayes classifier
nb = NaiveBayes(X_train, y_train)
# nb.fit(X_train, y_train)  # Not needed, training is done in __init__

# Make predictions on the test set
predictions = nb.predict(X_test)

# Print the classification accuracy
print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
