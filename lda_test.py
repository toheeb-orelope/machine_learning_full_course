import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from lda import LDA

data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the first two linear discriminants
lda = LDA(n_components=2)
lda.fit(X, y)
X_projected = lda.transform(X)
print(X_projected.shape)
print("Shape of transformed X", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.colorbar()
plt.show()
