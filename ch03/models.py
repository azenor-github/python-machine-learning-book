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


class DecisionTreeClassifier:

    def __init__(
        self, max_depth=float("inf"), min_samples_split=2, min_samples_leaf=1,
        min_impurity_decrease=0, criterion="entropy", precision=2
    ):
        self.config = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            precision=precision
        )
    
    def fit(self, X, y):
        self.tree_ = self.grow_tree(X, y)
        # self.test_termination_conditions

    def grow_tree(self, X, y, depth=0):
        if not self.test_max_depth(depth):
            return

        feature_index, split_value = self.determine_sample_split(X, y)

        if feature_index is None:
            return None

        X_l, X_r, y_l, y_r = self.split_sample(X, y, feature_index, split_value)

        node = dict(
            depth=depth,
            feature=feature_index,
            split_value=split_value,
            n=X.shape[0],
            yhat=self.count_classes(y)
        )
        node["left_node"] = self.grow_tree(X_l, y_l, depth+1)
        node["right_node"] = self.grow_tree(X_r, y_r, depth+1)
        return node

    def test_max_depth(self, depth):
        if depth > self.config["max_depth"]:
            return False
        return True

    def count_classes(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def split_sample(self, X, y, feature, value):
        X_l, X_r = X[X[:,feature] <= value,:], X[X[:,feature] > value,:]
        y_l, y_r = y[X[:,feature] <= value], y[X[:,feature] > value]
        return X_l, X_r, y_l, y_r

    def determine_sample_split(self, X, y):
        min_impurity_decrease = float("-inf")
        feature_index = None
        split_value = None

        for i in range(X.shape[1]):
            imp_dec, split = self.determine_feature_split(X[:,i], y)
            if imp_dec is not None and imp_dec > min_impurity_decrease:
                min_impurity_decrease = imp_dec
                feature_index = i
                split_value = split

        if not self.test_min_impurity_decrease(min_impurity_decrease):
            return None, None

        return feature_index, split_value

    def test_min_impurity_decrease(self, imp_dec):
        if imp_dec >= self.config["min_impurity_decrease"]:
            return True
        return False 

    def determine_feature_split(self, x, y):
        if not self.test_min_samples_split(x):
            return None, None
        splits = self.identify_potential_splits(x)
        return self.choose_the_best_split(x, y, splits)

    def test_min_samples_split(self, x):
        if x.shape[0] >= self.config["min_samples_split"]:
            return True
        return False

    def identify_potential_splits(self, x):
        ux = np.sort(np.unique(np.round(x, self.config["precision"])))
        return (ux[1:] + ux[:-1])/2

    def choose_the_best_split(self, x, y, splits):
        min_ig = float("-inf")
        split_value = None
        for split in splits:
            current_ig = self.calc_relative_information_gatin(x, y, split)
            if current_ig is not None and current_ig > min_ig:
                min_ig = current_ig
                split_value = split
        return min_ig, split_value

    def calc_relative_information_gatin(self, x, y, split):
        ig = self.calc_information_gain(x, y, split)
        split_info = self.calc_split_info(x, split)
        return ig and ig/split_info

    def calc_information_gain(self, x, y, split):
        x_l, x_r = x[x <= split], x[x > split]
        y_l, y_r = y[x <= split], y[x > split]

        if not self.test_min_samples_leaf(x_l, x_r):
            return None

        inp_input = self.calc_criterion(x, y)
        inp_left = self.calc_criterion(x_l, y_l)
        inp_right = self.calc_criterion(x_r, y_r)

        ig = inp_input - (
            (x_l.shape[0]/x.shape[0])*inp_left + 
            (x_r.shape[0]/x.shape[0])*inp_right
        )

        return ig

    def calc_split_info(self, x, split):
        x_l, x_r = x[x <= split], x[x > split]
        counts = np.array([x_l.shape[0], x_r.shape[0]])
        relcounts = counts/np.sum(counts)
        return np.sum(-relcounts * np.log2(relcounts))

    def calc_criterion(self, x, y):
        crit_function = self.get_criterion_function()
        return crit_function(x, y)

    def get_criterion_function(self):
        return getattr(self, "calc_%s" % self.config["criterion"], self.calc_entropy)

    def calc_entropy(self, x, y):
        data = np.array([ y[y == c].shape[0]/y.shape[0] for c in np.unique(y) ])
        return -np.sum(data*np.log2(data))

    def calc_gini(self, x, y):
        data = np.array([ y[y == c].shape[0]/y.shape[0] for c in np.unique(y) ])
        return np.sum(data*(1-data))

    def test_min_samples_leaf(self, x_l, x_r):
        if x_l.shape[0] < self.config["min_samples_leaf"]:
            return False
        if x_r.shape[0] < self.config["min_samples_leaf"]:
            return False
        return True

    def predict(self, X):
        return np.apply_along_axis(self.predict_class, axis=1, arr=X)

    def predict_class(self, x, node=None):
        node = node or self.tree_

        if node["left_node"] is not None \
                and x[node["feature"]] <= node["split_value"]:
            return self.predict_class(x, node["left_node"])

        if node["right_node"] is not None \
                and x[node["feature"]] > node["split_value"]:
            return self.predict_class(x, node["right_node"])

        return max(node["yhat"].items(), key=lambda x: x[1])[0]


class KNeighborsClassifier(object):

    def __init__(self, n_neighbors=3, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        dist = self.calc_distance_matrix(X)
        return self.make_predictions(dist)

    def calc_distance_matrix(self, X):
        return np.apply_along_axis(self.calc_distance, axis=1, arr=X).T

    def calc_distance(self, x):
        return np.sqrt(np.sum((self.X - x)**self.p, axis=1))

    def make_predictions(self, dist):
        return np.apply_along_axis(self.make_prediction, axis=0, arr=dist)

    def make_prediction(self, disti):
        dist_order_indexes = np.argsort(disti)
        y_neighbors = self.y[dist_order_indexes][:self.n_neighbors]
        return np.argmax(np.bincount(y_neighbors))