import numpy as np

from sklearn.base import RegressorMixin
from tree import DecisionTreeRegressor


class BaseGradientBoostingRegressor:

    def __init__(self,
                 n_estimators: 5,
                 max_depth: int,
                 learning_rate=None):

        self.estimator = DecisionTreeRegressor
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = [learning_rate] * n_estimators

        self._estimator_params = self.estimator().get_params()

        self._set_estimators()

    def _set_estimators(self):
        est_params = self._estimator_params

        self._estimators = [self.estimator(**est_params)
                            for _ in range(self.n_estimators)]

    def fit(self, X, y):

        y_pred = np.zeros_like(y)
        self._init_pred = np.mean(y)
        y_pred.fill(self._init_pred)
        y_true = y

        for i, est in enumerate(self._estimators):
            y_err = y_true - y_pred
            est.fit(X, y_err)
            yi_pred = est.predict(X)
            y_pred += self.learning_rate[i] * yi_pred

    def predict(self, X):
        y_pred = np.zeros(shape=(X.shape[0], 1)) + self._init_pred
        for i, est in enumerate(self._estimators):
            y_pred += self.learning_rate[i] * est.predict(X)
        return y_pred


class GradientBoostingRegressor(BaseGradientBoostingRegressor, RegressorMixin):

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 learning_rate=None):

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )


if __name__ == '__main__':

    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingRegressor2

    params = {'n_estimators': 5, 'max_depth': 2,
              'learning_rate': 1., 'loss': 'ls'}
    gb2 = GradientBoostingRegressor2(**params)

    gb = GradientBoostingRegressor(n_estimators=5, max_depth=2, learning_rate=1.)

    x = np.arange(0, 50).reshape(-1, 1)
    y1 = np.random.uniform(10, 15, 10)
    y2 = np.random.uniform(20, 25, 10)
    y3 = np.random.uniform(0, 5, 10)
    y4 = np.random.uniform(30, 32, 10)
    y5 = np.random.uniform(13, 17, 10)

    y = np.concatenate((y1, y2, y3, y4, y5))
    y = y[:, None]

    X_sh, y_sh = shuffle(x, y)

    gb.fit(x, y)
    gb2.fit(x, y)
    print("Friedman: ", gb.score(x, y))
    print("Friedman: ", gb2.score(x, y))
    x_cont = np.linspace(0, 50, 250).reshape(-1, 1)
    y_pred = gb.predict(x_cont)
    y_pred2 = gb2.predict(x_cont)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax[0].plot(x, y, 'o')
    ax[0].plot(x_cont, y_pred, 'r')
    ax[0].plot(x_cont, y_pred2, 'orange')

    ax[1].plot(x, y.ravel() - gb.predict(x).ravel(), 'o')
    plt.show()
