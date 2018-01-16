import numpy as np


class WeightsMixin(object):

    def init_weights(self, n, random_state=None):
        rgen = np.random.RandomState(random_state or self.random_state)
        self.w_ = rgen.normal(loc=0.03, scale=0.01, size=n)
        return self

    @property
    def weights(self):
        return getattr(self, "w_", None)


class LogisticRegresionGD(WeightsMixin):

    def __init__(self, eta=0.1, n_iter=10, lambda_=0.0, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.random_state = random_state

    def fit(self, X, y):
        self.init_weights(X.shape[1] + 1, self.random_state)
        self.cost_ = [
            self.update_weights(X, y)
            for _ in range(self.n_iter)
        ]
        return self

    def update_weights(self, X, y):
        yhat = self.activation(self.net_input(X))
        self.w_ -= self.eta * self.calc_gradient(X, y, yhat)
        return self.calc_cost(y, yhat)

    def calc_gradient(self, X, y, output):
        errors = y - output
        return np.hstack((-errors.sum(), -X.T.dot(errors) + self.lambda_*self.w_[1:]))

    def calc_cost(self, y, output):
        return -y.dot(np.log(output)) - (1-y).dot(np.log(1-output)) + self.lambda_*self.w_[1:].T.dot(self.w_[1:])

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) > 0.5, 1, 0)