# Importing libraries
import os
import csv
import numpy as np
import pandas as pd

# Importing the dataset
this_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(this_dir, "spambase.data")


def open_dataset(data_file):
    # skip_header=0, missing_values="---", filling_values=0.0
    data = np.genfromtxt(data_file, delimiter=",", dtype=np.float32)
    return data


# Option 1: Using np.array
"""
data = np.array(data, dtype=np.float32)
print(data.shape)
"""
# Option 2: Using numpy.loadtxt
"""
data = np.loadtxt(DATASET_FILE, delimiter=",", dtype=np.float32)
print(data.shape, data.dtype)
"""

# Options 3: Using pandas.read_csv
# na_values = ['---', '']
"""
df = pd.read_csv(DATASET_FILE, delimiter=",", header=None, skiprows=0, dtype=np.float32)
df = df.fillna(0.0)

# df to numpy array
data = df.to_numpy()
print(data[4, 0:5])

# convert datatypes in numpy
# data = np.asarray(data, dtype = np.float32)
# print(data.dtype)
"""

# Option 4: Using numpy.genfromtxt
# Skip the first row (header), fill missing values with "---", fill empty values with 0.0
# data = np.genfromtxt(DATASET_FILE, delimiter=",", dtype=np.float32)
# print(data.shape, data.dtype)


# Splitting the dataset into the Training set and Test set
def split_dataset(data):
    n_samples, n_features = data.shape
    n_features -= 1  # Exclude the last column which is the label

    X = data[:, 0:n_features]
    y = data[:, n_features]

    return X, y
