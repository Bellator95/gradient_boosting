import numpy as np
from abc import ABCMeta

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

from tree_builder import TreeBuilder
from utils import most_frequent_class


class Tree(BaseEstimator, metaclass=ABCMeta):
    """ Base class for the tree estimators.

    Arguments
    ---------
        merge_leaf_func: function,
            The method of merging elements in leave to scalar.

        criterion: string,
            The name of criterion to split data.

        max_depth: integer,
            The maximal depth of tree to build.
    """

    def __init__(self,
                 merge_leaf_func,
                 criterion,
                 max_depth=None):

        self.tree_ = None
        self.feature_importances_ = None
        self.output_shape_ = None

        self._tree_builder = TreeBuilder(
            merge_leaf_func=merge_leaf_func,
            criterion=criterion,
            max_depth=max_depth
        )

    def fit(self, X, y):
        """ Build tree with respect to training data.

        Parameters
        ----------
            X : array, shape (n_samples, n_features)
                Training data

            y : array, shape (n_samples,)
                Target values.
        """
        self.output_shape_ = y.shape
        self.tree_ = self._tree_builder.build(X, y)
        self.feature_importances_ = self._tree_builder.feature_importances_

    # TODO: To refactor method to avoid using recursion.
    def _walk_tree(self, node, data, indexes):
        """ Passes data to the leaves of the tree and returns the node labels. """

        if node.leaf_value is not None:
            return np.zeros(shape=(data.shape[0])) + node.leaf_value, indexes

        data_left, indexes_left, data_right, indexes_right = \
            node.splitter.split(data, indexes)

        labels_left, indexes_left = self._walk_tree(
            node.left_node, data_left, indexes_left)
        labels_right, indexes_right = self._walk_tree(
            node.right_node, data_right, indexes_right)

        labels = np.concatenate([labels_left, labels_right])
        indexes = np.concatenate([indexes_left, indexes_right])

        return labels, indexes

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
            y : array, shape (n_samples,)
                Returns predicted values.
        """

        labels, indexes = self._walk_tree(self.tree_, X, np.arange(0, X.shape[0]))
        return labels[np.argsort(indexes)].reshape(-1, *self.output_shape_[1:])

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class DecisionTreeClassifier(Tree, ClassifierMixin):
    """ Class defines a binary decision tree classifier.

    Arguments
    ---------
        criterion : string, optional (default="gini")
            The name of the criterion which should be used during the tree building.

        max_depth : integer, optional (default=None)
            The maximal depth of the decision tree.

    """
    def __init__(self, criterion="gini", max_depth=None):
        super().__init__(
            merge_leaf_func=most_frequent_class,
            criterion=criterion,
            max_depth=max_depth)

        self.criterion = criterion
        self.max_depth = max_depth

        self.n_classes_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        super().fit(X, y)


class DecisionTreeRegressor(Tree, RegressorMixin):
    """ Class defines a binary decision tree regressor.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The name of the criterion which should be used during the tree building.

    max_depth : integer, optional (default=None)
        The maximal depth of the decision tree.

    """
    def __init__(self, criterion="friedman_mse", max_depth=None):
        super().__init__(
            merge_leaf_func=np.median,
            criterion=criterion,
            max_depth=max_depth)

        self.criterion = criterion
        self.max_depth = max_depth


if __name__ == "__main__":
    from sklearn.utils import shuffle

    N1 = N2 = N3 = 100
    X1 = np.random.normal(loc=0, scale=2, size=(N1, 2))
    y1 = np.zeros(shape=(N1,))
    X2 = np.random.normal(loc=[2, 2], scale=2, size=(N2, 2))
    y2 = np.ones(shape=(N2,))
    X3 = np.random.normal(loc=[4, 4], scale=1, size=(N3, 2))
    y3 = np.ones(shape=(N3,)) + 1

    X = np.concatenate([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    X_sh, y_sh = shuffle(X, y)


    def get_grid(data):
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


    xx, yy = get_grid(X_sh)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf.fit(X_sh, y_sh)
    print(clf.score(X_sh, y_sh))

    predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    import matplotlib.pyplot as plt

    plt.pcolormesh(xx, yy, predictions)
    plt.scatter(X_sh[:, 0], X_sh[:, 1], c=y_sh, s=100,
                edgecolors='black', linewidth=1.5)
    plt.show()

    x = np.arange(0, 50).reshape(-1, 1)
    y1 = np.random.uniform(10, 15, 10)
    y2 = np.random.uniform(20, 25, 10)
    y3 = np.random.uniform(0, 5, 10)
    y4 = np.random.uniform(30, 32, 10)
    y5 = np.random.uniform(13, 17, 10)

    y = np.concatenate((y1, y2, y3, y4, y5))
    y = y[:, None]

    X_sh, y_sh = shuffle(x, y)

    clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=3)
    clf.fit(X_sh, y_sh)
    print("Friedman: ", clf.score(x, y))
    x_cont = np.linspace(0, 50, 250).reshape(-1, 1)
    y_pred = clf.predict(x_cont)

    plt.plot(x, y, 'o')
    plt.plot(x_cont, y_pred, 'r')
    plt.show()
