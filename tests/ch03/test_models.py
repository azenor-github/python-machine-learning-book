import unittest
from unittest import mock

import numpy as np

import ch03.models
from ch03.models import *


class LogisticRegresionTest(unittest.TestCase):

    def test_net_input(self):
        lr = LogisticRegresionGD()
        lr.w_ = np.array([1.0, 1.0, 1.0])

        z = lr.net_input(np.array([[1, 1], [2, 2]]))

        self.assertTrue(np.array_equal(z, [3, 5]))

    def test_activation_function(self):
        lr = LogisticRegresionGD()

        y = lr.activation(np.array([0, 10]))

        self.assertTrue(np.array_equal(y, [0.5, 1./(1. + np.exp(-10))]))

    def test_calculate_gradient(self):
        lr = LogisticRegresionGD(n_iter=2, eta=1)
        lr.w_ = np.array([1.0, 1.0, 1.0])

        X = np.array([[0, 1], [1, 0]])
        y = np.array([1, 0])

        output = lr.activation(lr.net_input(X))
        gradient = lr.calc_gradient(X, y, output)

        self.assertTrue(np.allclose(
            gradient, np.array([ 0.76159416,  0.88079708, -0.11920292]),
            atol=0.01
        ))

    def test_calculate_gradient_with_regularization_panetly(self):
        lr = LogisticRegresionGD(n_iter=2, eta=1, lambda_=0.5)
        lr.w_ = np.array([1.0, 1.0, 1.0])

        X = np.array([[0, 1], [1, 0]])
        y = np.array([1, 0])

        output = lr.activation(lr.net_input(X))
        gradient = lr.calc_gradient(X, y, output)

        self.assertTrue(np.allclose(
            gradient, 
            np.array([ 0.76159416,  0.88079708+0.5, -0.11920292+0.5]),
            atol=0.01
        ))

    def test_update_weights_in_one_iteration(self):
        lr = LogisticRegresionGD(n_iter=2, eta=1)
        lr.w_ = np.array([1.0, 1.0, 1.0])

        X = np.array([[0, 1], [1, 0]])
        y = np.array([1, 0])

        lr.update_weights(X, y)

        self.assertTrue(np.allclose(
            lr.weights, np.array([ 0.23840584,  0.11920292,  1.11920292]),
            atol=0.01
        ))

    def test_calc_cost(self):
        lr = LogisticRegresionGD(n_iter=2, eta=1)
        lr.w_ = np.array([1, 1])

        cost = lr.calc_cost(np.array([1, 0]), np.array([0.75, 0.25]))

        self.assertEqual(cost, -2*np.log(0.75))

    @mock.patch("ch03.models.LogisticRegresionGD.init_weights")
    def test_fit_model_to_data(self, iw_mock):
        lr = LogisticRegresionGD(n_iter=2, eta=1)
        lr.w_ = np.array([1.0, 1.0, 1.0])

        X = np.array([[0, 1], [1, 0]])
        y = np.array([1, 0])

        lr.fit(X, y)

        self.assertTrue(np.allclose(
            lr.weights, np.array([-0.1454264 , -0.46925854,  1.32383214]),
            atol=0.01
        ))

    @mock.patch("ch03.models.LogisticRegresionGD.init_weights")
    def test_fit_method_saves_cost_of_learning_from_every_iteration(self, iv_mock):
        lr = LogisticRegresionGD(n_iter=2, eta=1)
        lr.w_ = np.array([1.0, 1.0, 1.0])

        X = np.array([[0, 1], [1, 0]])
        y = np.array([1, 0])

        lr.fit(X, y)

        self.assertTrue(hasattr(lr, "cost_"))

    @mock.patch("ch03.models.LogisticRegresionGD.activation")
    def test_predict_final_class(self, act_mock):
        act_mock.return_value = np.array([0.25, 0.75])

        lr = LogisticRegresionGD(n_iter=2, eta=1)
        lr.w_ = np.array([1, 1, 1])

        yhat = lr.predict(np.array([[0, 1], [1, 0]]))

        self.assertTrue(np.array_equal(yhat, [0, 1]))

    def test_increase_cost_in_L2_regularization(self):
        lr = LogisticRegresionGD(n_iter=2, eta=1, lambda_=1.0)
        lr.w_ = np.array([1.0, 2.0])

        cost = lr.calc_cost(np.array([1, 0]), np.array([0.75, 0.25]))

        self.assertEqual(cost, -2*np.log(0.75) + 4)

