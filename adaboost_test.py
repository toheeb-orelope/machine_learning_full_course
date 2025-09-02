import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adaboost import Adaboost


# Function to calculate accuracy of predictions
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Load the breast cancer dataset from scikit-learn
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Convert class labels from 0/1 to -1/1 for Adaboost
y[y == 0] = -1

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Initialize Adaboost classifier with 5 weak classifiers
clf = Adaboost(n_clf=5)
# Train the classifier
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)
