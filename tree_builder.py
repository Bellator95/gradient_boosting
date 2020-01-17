import numpy as np

from splitter import Splitter
from queue import LifoQueue


class _Node():
    """ Class represents node data structure in tree.

    Arguments
    ---------
        leaf_value: [int, float], optional (default=None),
            The value in leaf node.

        splitter: Splitter, optional (default=None),
            The fitted splitter.
    """
    def __init__(self, leaf_value=None, splitter=None):
        self.leaf_value = leaf_value
        self.splitter = splitter

        self.left_node = None
        self.right_node = None

    def reset_nodes(self):
        """ Reset left and right node to the empty one. """
        self.left_node = _Node()
        self.right_node = _Node()


class TreeBuilder():
    """ Builder class for tree data structure.

    The strategy of building is depth-first.

    Arguments
    ---------
        merge_leaf_func: function,
            The function which is used for merging predictions in leaf node into single scalar.

        criterion: string,
            The name of criterion to split data and reduce impurity.

        max_depth: integer,
            The maximal depth of the constructed tree.
    """

    def __init__(self,
                 merge_leaf_func,
                 criterion,
                 max_depth):

        self.merge_leaf_func = merge_leaf_func
        self.criterion = criterion
        self.max_depth = max_depth

        self.feature_importances_ = None

    def build(self, X, y):
        """ Searches in X using y for a feature and a threshold
            to split X into two disjoint sets.

        Parameters
        ----------
            X : array, shape (n_samples, n_features)
                Training data

            y : array, shape (n_samples,)
                Target values.
        """

        self.feature_importances_ = np.zeros(shape=(X.shape[1],), dtype=np.float32)

        maxsize = 0
        if self.max_depth is not None:
            maxsize = 2 * self.max_depth + 1

        stack = LifoQueue(maxsize)
        indexes = np.arange(y.shape[0])
        depth = 0

        root = _Node()
        stack.put([root, indexes, depth])

        while stack.qsize():
            node, indexes, depth = stack.get()

            if depth == self.max_depth:
                node.leaf_value = self.merge_leaf_func(y[indexes])
                continue

            splitter = Splitter(self.criterion)
            left_indexes, right_indexes = splitter.fit_split(
                X[indexes], y[indexes], only_indexes=True)

            self.feature_importances_[splitter.feature_id_] += splitter.impurity_reduction_

            if left_indexes.size == 0 or right_indexes.size == 0:
                node.leaf_value = self.merge_leaf_func(y[indexes])
            else:
                left_indexes = indexes[left_indexes]
                right_indexes = indexes[right_indexes]

                node.splitter = splitter
                node.reset_nodes()

                stack.put([node.right_node, right_indexes, depth + 1])
                stack.put([node.left_node, left_indexes, depth + 1])

        self.feature_importances_ /= self.feature_importances_.sum()
        return root
