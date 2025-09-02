import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nb import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, n_informative=5, random_state=123
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

nb = NaiveBayes(X_train, y_train)
# nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
