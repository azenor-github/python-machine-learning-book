import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier #model for tests

from ch04.feature_selection import FFS


class FFSTest(unittest.TestCase):
    
    def setUp(self):
        self.X, self.y = self.create_test_dataset(0)

    def create_test_dataset(self, seed=None):
        np.random.seed(seed)
        x1 = np.hstack([ # the most predictive feature
            np.random.normal(loc=3, scale=1, size=10),
            np.random.normal(loc=-3, scale=1, size=10)
        ])
        x2 = np.hstack([ # the second most predictive feature
            np.random.normal(loc=2, scale=1, size=10),
            np.random.normal(loc=-2, scale=1, size=10)
        ])
        x3 = np.random.normal(loc=0, scale=1, size=20) # random feature

        X = np.array([x1, x2, x3]).T
        y = np.concatenate([np.zeros(10), np.ones(10)])

        return X, y

    def test_select_the_most_predictive_feature(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        ffs = FFS(knn, k_features=1).fit(self.X, self.y)

        self.assertEqual(len(ffs.indices_), 1)
        self.assertEqual(ffs.indices_[0], 0)

    def test_select_model_with_two_the_most_predictive_features(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        ffs = FFS(knn, k_features=2).fit(self.X, self.y)

        self.assertEqual(len(ffs.indices_), 2)
        self.assertEqual(ffs.indices_[0], 0)
        self.assertEqual(ffs.indices_[1], 1)

    def test_for_selecting_more_features_than_in_dataset(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        ffs = FFS(knn, k_features=10).fit(self.X, self.y)

        self.assertEqual(len(ffs.indices_), 3)

    def test_for_selecting_all_features(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        ffs = FFS(knn).fit(self.X, self.y)

        self.assertEqual(len(ffs.indices_), 3)