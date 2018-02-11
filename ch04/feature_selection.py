from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class EFS(object):

    def __init__(
        self, estimator, k_features=float("Inf"), scoring=accuracy_score
    ):
        self.k_features = k_features
        self.estimator = estimator
        self.scoring = scoring
        self.indices_ = []      

    def fit(self, X, y):
        self.subsets_ = [ 
            self.find_the_best_subset(X, y, k) 
            for k in self.get_features_range(X)
        ]
        return self

    def find_the_best_subset(self, X, y, k):
        subsets = list(combinations(range(X.shape[1]), k))
        scores = [ self.calc_score(X, y, indices) for indices in subsets ]
        return subsets[np.argmax(scores)]

    def calc_score(self, X, y, indices):
        self.estimator.fit(X[:, indices], y)
        y_pred = self.estimator.predict(X[:, indices])
        score = self.scoring(y, y_pred)
        return score

    def get_features_range(self, X):
        return range(1, min(X.shape[1], self.k_features)+1)


class FFS(object):

    def __init__(self, estimator, k_features=float("Inf"), scoring=accuracy_score):
        self.k_features = k_features
        self.estimator = estimator
        self.scoring = scoring
        self.indices_ = []

    def fit(self, X, y):
        self.indices_ = [] 
        self.subsets_ = []

        while self.is_termination_condition():
            feature_index = self.select_next_feature(X, y, self.indices_)
            if feature_index is None:
                break
            self.indices_.append(feature_index)
            self.subsets_.append(list(self.indices_))

        return self

    def is_termination_condition(self):
        return len(self.indices_) < self.k_features

    def select_next_feature(self, X, y, selected):
        features = list(set(range(X.shape[1])) - set(selected))
        scores = [
            self.calc_score(X, y, selected + [feature])
            for feature in features
        ]
        if len(scores) > 0:
            return features[np.argmax(scores)]
        else:
            return None

    def calc_score(self, X, y, indices):
        self.estimator.fit(X[:, indices], y)
        y_pred = self.estimator.predict(X[:, indices])
        score = self.scoring(y, y_pred)
        return score


class SBS():
    
    def __init__(
        self, estimator, k_features, scoring=accuracy_score, 
        test_size=0.25, random_state=1
    ):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        dim = X_train.shape[1] # number of features
        self.indices_ = tuple(range(dim)) # (0, 1, 2, ...)
        self.subsets_ = [ self.indices_ ]
        
        # calculate score for full model
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(
                    X_train, y_train, X_test, y_test, p
                )
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)            
            self.scores_.append(scores[best])
            
            dim -= 1
        
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score