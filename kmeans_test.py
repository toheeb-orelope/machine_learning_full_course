import numpy as np
from sklearn import datasets
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kmeans import KMeans

# Generate synthetic data with 4 clusters
X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X.shape)  # Print shape of data

clusters = len(np.unique(y))  # Number of clusters
print(clusters)

# Initialize KMeans with number of clusters and fit to data
k = KMeans(K=clusters, max_iters=150, plot_steps=False)
y_pred = k.predict(X)  # Predict cluster assignments

k.plot()  # Visualize the clusters and centroids
