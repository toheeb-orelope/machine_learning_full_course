# CVS
# NumPy, 2X
# Pandas

# Missing data, header, dtype


import csv
import numpy as np
import pandas as pd

FILE_NAME = "spambase.data"

# Load data from CSV file using pure Python
"""
with open(FILE_NAME, "r") as file:
    data = list(csv.reader(file, delimiter=","))

data = np.array(data)
print(data.shape)
"""

# Load data from CSV file using NumPy
# data = np.loadtxt(FILE_NAME, delimiter=",")
# skip_header=1 if there is a header row to avoid errors
# dtype=np.float32 to save memory
# Missing values can be handled with genfromtxt
# fill missing values with 0.0
data = np.genfromtxt(FILE_NAME, delimiter=",", dtype=np.float32, skip_header=1, missing_values=np.nan, filling_values=0.0)

print(data.shape, data.dtype, type(data[0][0]))
n_samples, n_features = data.shape
n_features -= 1  # last column is the label
X = data[:, 0:n_features]
y = data[:, n_features]
print(X.shape, y.shape)
print(X[0, 0:5])


# Load data from CSV file using Pandas
# skiprows=1 if there is a header row to avoid errors
# dtype=np.float32 to save memory
# Missing values are automatically handled as NaN
df = pd.read_csv(FILE_NAME, delimiter=",", header=None, dtype=np.float32, skiprows=1, na_values=np.nan)
data = df.to_numpy()
data = np.asarray(data, dtype=np.float32)  # Convert data to float32
n_samples, n_features = data.shape
n_features -= 1  # last column is the label
X = data[:, 0:n_features]
y = data[:, n_features]
print(X.shape, y.shape)
print(X[0, 0:5])  # view the first 5 features of the first sample
