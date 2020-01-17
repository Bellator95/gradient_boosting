import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractCriterion(metaclass=ABCMeta):

    @abstractmethod
    def impurity_score(self, y):
        """ Calculates impurity score for the target values.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.
        """

    def split_impurity_score(self, y_left, y_right):
        """ Computes the impurity score of split.

        Parameters
        ----------
            y_left: array, shape (n_samples,)
                    The target values in left split.

            y_right: array, shape (n_samples,)
                    The target values in left split.
        """

        n_left, n_right = y_left.shape[0], y_right.shape[0]
        n_root = n_left + n_right

        right_impurity = self.impurity_score(y_right)
        left_impurity = self.impurity_score(y_left)
        return (n_right / n_root) * right_impurity + (n_left / n_root) * left_impurity

    def impurity_improvement(self, y_left, y_right):
        """ Computes the impurity improvement for split.

        Parameters
        ----------
            y_left: array, shape (n_samples,)
                    The target values in left split.

            y_right: array, shape (n_samples,)
                    The target values in left split.
        """
        y = np.r_[y_left, y_right]

        initial_score = self.impurity_score(y)
        split_score = self.split_impurity_score(y_left, y_right)
        return initial_score - split_score


class GiniCriterion(AbstractCriterion):
    """ Gini criterion for classification. """

    def impurity_score(self, y):
        """ Returns Gini impurity score. """
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.power(counts / y.shape[0], 2).sum()


class EntropyCriterion(AbstractCriterion):
    """ Entropy criterion for classification. """

    def impurity_score(self, y):
        """ Returns Entropy impurity score. """
        _, counts = np.unique(y, return_counts=True)
        p = counts / y.shape[0]
        return - p @ np.log2(p)


class MSECriterion(AbstractCriterion, metaclass=ABCMeta):
    """ MSE criterion for classification. """

    def impurity_score(self, y):
        return np.var(y)


class FriedmanMSECriterion(MSECriterion):
    """ Friedman MSE criterion for classification. """

    def split_impurity_score(self, y_left, y_right):
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        return ((n_left * n_right) / (n_left + n_right)) * (np.mean(y_left) - np.mean(y_right))**2

    impurity_improvement = split_impurity_score


CRITERIONS = {
    "gini": GiniCriterion(),
    "entropy": EntropyCriterion(),

    "mse": MSECriterion(),
    "friedman_mse": FriedmanMSECriterion(),
}
