import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from linear_regression import LinearRegression

# Generate a synthetic regression dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Uncomment to visualize the raw data
# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color='b', marker='o', s=30)
# plt.show()

# Initialize and train the Linear Regression model
regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)

# Make predictions on the test set
predictions = regressor.predict(X_test)


# Define Mean Squared Error function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Calculate and print the MSE for test predictions
mse_value = mse(y_test, predictions)
print("MSE:", mse_value)

# Predict values for all data points to plot the regression line
y_pred_line = regressor.predict(X)

# Set up color map for plotting
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))

# Plot training data points
m1 = plt.scatter(X_train.flatten(), y_train, color=cmap(0.9), s=10)
# Plot test data points
m2 = plt.scatter(X_test.flatten(), y_test, color=cmap(0.5), s=10)
# Plot the regression prediction line
plt.plot(X.flatten(), y_pred_line, color="black", linewidth=2, label="Prediction Line")
plt.show()
