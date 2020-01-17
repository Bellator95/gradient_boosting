import numpy as np
from criterions import CRITERIONS


class Splitter():
    """ 
    The splitter searches in the input space for a feature and
    a threshold to split data into two disjoint sets.
    
    Parameters
    ----------
    criterion : string,
        The criterion name to measure the quality of the split.
            
    n_float_blocks : int, optional (default=100)
        The number of blocks into which the float type feature will be divided
        if it contains more values than n_float_blocks.
            
    n_int_blocks : int, optional (default=100)
        The number of blocks into which integer type feature will be divided
        if it contains more unique values than n_int_blocks.
            
    Attributes
    ----------
    threshold_ : float, int
        Optimal threshold to split the input data on the two disjoint sets.
        
    feature_id_ : int
        The feature index by which split is done.
        
    impurity_reduction_ : float
        The impurity reduction after the split by feature[feature_id_] > threshold_
        
    Examples
    --------
        
    >>> splitter = Splitter(criterion="gini")
    >>> X = np.array([
    ...     [1, 10, 100],
    ...     [5, 9, 98],
    ...     [4, 11, 10],
    ...     [50, 40, 17],
    ...     [77, 53, 12],
    ...     [3, 50, 96],
    ... ])
    >>> y = np.array([0, 0, 0, 0, 0, 1])
    >>> X_left, y_left, X_right, y_right = splitter.fit_split(X, y)
    >>> X_left
    array([[  1,  10, 100],
            [  3,  50,  96]])
    >>> X_right
    array([[ 5,  9, 98],
            [ 4, 11, 10],
            [50, 40, 17],
            [77, 53, 12]])
    >>> splitter.threshold_
    3
    >>> splitter.feature_id_
    0
    >>> splitter.impurity_reduction_
    0.11111111111111102
    
    """
    def __init__(self, criterion,
                 n_float_blocks=100,
                 n_int_blocks=100):
        
        self._criterion = CRITERIONS[criterion]
        self._n_float_blocks = n_float_blocks
        self._n_int_blocks = n_int_blocks
        self._max_n_blocks = max(self._n_float_blocks, self._n_int_blocks)
        self._impurity_improvement = np.zeros(shape=(self._max_n_blocks,))
        
        self.threshold_ = None
        self.feature_id_ = 0
        self.impurity_reduction_ = None
        
        self._root_impurity = None
        
    def _get_feature_values(self, feature):
        """ Returns the selected feature values for further search. 

        Parameters
        ----------
            feature: array, shape  (n_samples,).
        """
        if feature.shape[0] > self._max_n_blocks:
            min_val, max_val = np.min(feature), np.max(feature)

            if self._has_int_type(feature):
                feature = feature.astype(np.int32)
                feature_values = np.unique(feature)

                if feature_values.shape[0] < self._n_int_blocks:
                    feature_values = np.sort(feature_values)
                else:
                    feature_values = np.linspace(min_val, max_val, self._n_int_blocks)
            else:
                feature_values = np.linspace(min_val, max_val, self._n_float_blocks)
        else:
            feature_values = np.sort(np.unique(feature))
        return feature_values[:-1]
        
    def fit(self, X, y):
        """ Searches in X using y for a feature and a threshold
            to split X into two disjoint sets.

        Parameters
        ----------
            X : array, shape (n_samples, n_features)
                Training data
            y : array, shape (n_samples,)
                Target values.
        """

        self._root_impurity = self._criterion.impurity_score(y)

        if np.isclose(self._root_impurity, 0.):
            return

        self.impurity_reduction_ = - np.inf
        
        for feature_id, feature in enumerate(X.T):
            feature_values = self._get_feature_values(feature)

            for i, threshold in enumerate(feature_values):
                right_indexes = feature > threshold
                left_indexes = np.logical_not(right_indexes)
                self._impurity_improvement[i] = self._criterion.impurity_improvement(
                    y[left_indexes], y[right_indexes])

            max_reduction_value_id = int(np.argmax(self._impurity_improvement))
            
            if self.impurity_reduction_ < self._impurity_improvement[max_reduction_value_id]:
                self.impurity_reduction_ = self._impurity_improvement[max_reduction_value_id]
                self.feature_id_ = feature_id
                self.threshold_ = feature_values[max_reduction_value_id]

            self._impurity_improvement.fill(0)
    
    def split(self, X, y=None, only_indexes=False):
        """ Splits data on the two disjoint sets.
        
        Parameters
        ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Training data

            y : array, shape (n_samples,), optional (default=None).
                Target values. Will be cast to X's dtype if necessary
                
            only_indexes: boolean, optional (default=False)
                If True, returns just indexes of split.

        Returns
        -------
           Returns split data or indexes.

        """
        if self._root_impurity is None:
            raise ValueError("Splitter is not fitted. Call function 'fit()' before 'split()'.")
            
        if np.isclose(self._root_impurity, 0.):
            if only_indexes:
                output = np.arange(y.shape[0]), np.array([])
            elif y is None:
                output = X, np.array([], dtype=X.dtype)
            else:
                output = X, y, np.array([], dtype=X.dtype), np.array([], dtype=y.dtype)
            return output
        
        feature = X[:, self.feature_id_]
        right_indexes_mask = feature > self.threshold_
        left_indexes_mask = np.logical_not(right_indexes_mask)

        if only_indexes:
            output = np.nonzero(left_indexes_mask)[0], np.nonzero(right_indexes_mask)[0]
        elif y is None:
            output = X[left_indexes_mask], X[right_indexes_mask]
        else:
            output = X[left_indexes_mask], y[left_indexes_mask], \
                     X[right_indexes_mask], y[right_indexes_mask]
        return output
        
    def fit_split(self, X, y, only_indexes=False):
        self.fit(X, y)
        return self.split(X, y, only_indexes)
        
    @staticmethod
    def _has_int_type(feature):
        """ Check if feature has integer data type.
        
        Parameters
        ----------
        feature : array, shape (n_samples,)
        
        Returns
        -------
            True if feature has integer data type else False.
            
        """
        return np.isclose(np.abs(feature.astype(np.int32) - feature).sum(), 0.)


if __name__ == "__main__":
    splitter = Splitter(criterion="gini")
    X = np.array([
        [1, 10, 100],
        [5, 9, 98],
        [4, 11, 10],
        [50, 40, 17],
        [77, 53, 12],
        [3, 50, 96],
        ])
    y = np.array([0, 0, 0, 0, 0, 1])
    X_left, y_left, X_right, y_right = splitter.fit_split(X, y)
    print(X_left)
    # array([[1, 10, 100],
    #        [3, 50, 96]])
    print(X_right)
    # array([[5, 9, 98],
    #        [4, 11, 10],
    #        [50, 40, 17],
    #        [77, 53, 12]])
    print(splitter.threshold_)
    # 3
    print(splitter.feature_id_)
    # 0
    print(splitter.impurity_reduction_)
    # 0.11111111111111102
