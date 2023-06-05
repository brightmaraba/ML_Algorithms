# K-nearest neighbors (KNN) classification
# Assigns the label of the most similar training example to a given input example
# Similarity is based on the Euclidean distance between the input example and the labeled examples.
# Eucledian distance: d(x, y) = sqrt(sum((x_i - y_i)^2))


class KNN:
    def __init__(self, k=3):  # k is the number of nearest neighbors - default is 3
        self.k = k

    def fit(self, X, y):  # X is the training data, y is the labels
        pass

    def predict(self, X):  # X is the test data
        pass
