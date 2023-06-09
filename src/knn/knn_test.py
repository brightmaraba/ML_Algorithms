import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

irirs = datasets.load_iris()
X, y = irirs.data, irirs.target
