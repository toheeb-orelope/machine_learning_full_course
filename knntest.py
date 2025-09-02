# Import required libraries for data handling and visualization
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define a color map for plotting (not used in this script, but useful for visualization)
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# Import the custom KNN classifier
from knn import KNN

# Load the Iris dataset (a classic multi-class classification dataset)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Create an instance of the KNN classifier with k=5 neighbors
clf = KNN(k=5)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the labels of the test data
predictions = clf.predict(X_test)

# Calculate and print the classification accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print("KNN classification accuracy:", accuracy)
