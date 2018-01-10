import operator

import numpy as np


class WeightsMixin(object):

    def init_weights(self, n, random_state=None):
        rgen = np.random.RandomState(random_state or self.random_state)
        self.w_ = rgen.normal(loc=0.03, scale=0.01, size=n)
        return self

    @property
    def weights(self):
        return getattr(self, "w_", None)

        
class Perceptron(WeightsMixin):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.init_weights(1 + X.shape[1])

        self.errors_ = []
        for _ in range(self.n_iter):
            self.errors_.append(self.update_weights(X, y))
        
        return self

    def update_weights(self, X, y):
        return sum(
            self.update_weights_for_single_sample(xi, target) 
            for xi, target in zip(X, y)
        )

    def update_weights_for_single_sample(self, xi, target):
        update = self.eta * (target - self.predict(xi))
        self.w_[1:] += update * xi
        self.w_[0] += update
        return int(update != 0.0)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


class AdalineGD(WeightsMixin):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.init_weights(1 + X.shape[1])
        self.cost_ = [ self.update_weights(X, y) for _ in range(self.n_iter) ]
        return self

    def update_weights(self, X, y):
        z = self.net_input(X)
        output = self.activation(z)
        errors = y - output
        self.w_[1:] += self.eta * X.T.dot(errors)
        self.w_[0] += self.eta * errors.sum()
        cost = 0.5 * (errors**2).sum()
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(WeightsMixin):
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.reval().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineMBGD(WeightsMixin):
    
    def __init__(
        self, eta=0.01, n_iter=10, batch=None, shuffle=True, random_state=None
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.batch = batch
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self.init_weights(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            self.cost_.append(self.update_weights(X, y))
        return self

    def update_weights(self, X, y):
        if self.shuffle:
            X, y = self._shuffle(X, y)
        batches = self._split_into_batches(X, y)
        cost = 0
        for b_X, b_y in batches:
            yhat = self.activation(self.net_input(b_X))
            errors = b_y - yhat
            self.w_[1:] += self.eta * b_X.T.dot(errors)
            self.w_[0] += self.eta * np.sum(errors)
            cost += sum(errors**2)
        return cost / X.shape[0]

    def _split_into_batches(self, X, y):
        splits = list(range(0, X.shape[0], self.batch)) + [None]
        return [
            (X[splits[index-1]:splits[index]], y[splits[index-1]:splits[index]])
            for index in range(1, len(splits))
        ]

    def _shuffle(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        r = rgen.permutation(len(y))
        return X[r], y[r]


    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class MultiClassClassifier:

    def __init__(self, clf, **kwargs):
        self.clf = clf
        self.kwargs = kwargs

    def fit(self, X, y):
        self.clfs_ = [
            {
                "cls": sample["cls"],
                "clf": self.clf(**self.kwargs).fit(sample["X"], sample["y"])
            }
            for sample in self.create_train_samples(X, y)
        ]
            
    def create_train_samples(self, X, y):
        return (
            dict(y=np.where(y == cls, -1, 1), X=X, cls=cls)
            for cls in np.unique(y)
        )

    def predict(self, X, method=operator.attrgetter("net_input")):
        if not hasattr(self, "clfs_"):
            raise RuntimeError(
                "The classifier has not been initialized. Call fit method first."
            )

        zs = [ method(clf["clf"])(X) for clf in self.clfs_ ]
        yhat = np.vstack(zs).T
        max_yhats = np.argmax(yhat, axis=1)

        return np.array([ self.clfs_[index]["cls"] for index in max_yhats ])