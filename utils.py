import numpy as np


def most_frequent_class(y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]
