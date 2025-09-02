import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from svm import SVM

# Generate a synthetic binary classification dataset
X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
# Convert labels to -1 and 1 for SVM
y = np.where(y <= 0, -1, 1)

# Initialize and train the SVM classifier
clf = SVM()
clf.fit(X, y)
# Print learned weights and bias
print(clf.w, clf.b)


# Function to visualize the SVM decision boundary and margins
def visualize_svm():
    def get_hyperplane_value(X, w, b, offset):
        # Calculate the y-value of the hyperplane for a given x-value
        return (-w[0] * X + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    # Get min and max values for x-axis
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # Calculate decision boundary and margins
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)
    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    # Plot decision boundary and margins
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    # Set plot limits
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])
    plt.show()


# Call the visualization function
visualize_svm()
