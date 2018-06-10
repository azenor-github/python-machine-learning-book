import unittest

import numpy as np

from ch12.neuralnet import *


class MLNNTest(unittest.TestCase):

    def create_net211_with_sigmoid_act_func(self):
        net = MLNN([2, 1, 1], act_func=sigmoid, eta=1)
        net.weights = [
            np.array([
                [0.5,],
                [0.1,],
                [0.2,]
            ]),
            np.array([
                [1], [0.3]
            ])
        ]
        return net

    def test_init_weights_for_single_perceptron_model(self):
        perceptron = MLNN([2, 1], act_func=linear, eta=1)

        self.assertEqual(len(perceptron.weights), 1)

        weights = perceptron.weights[0]
        self.assertCountEqual(weights.shape, (3, 1))

    def test_calc_gradients_for_single_perceptron_model_with_sigmoid(self):
        perceptron = MLNN([2, 1], act_func=sigmoid, eta=1)
        perceptron.weights = [ np.array([[0.5], [0.1], [0.2]]) ]

        X = np.array([
            [1, 1],
            [3, 2],
            [-1, 2]
        ])
        Y = np.array([[0], [1], [0]])

        ff_output = perceptron._feedforward(X)
        gradients = perceptron._calc_gradients(ff_output, Y)

        self.assertEqual(len(gradients), 1)
        self.assertCountEqual(gradients[0].shape, (3, 1))

        self.assertAlmostEqual(gradients[0][0][0], 0.25400630956647, 4)
        self.assertAlmostEqual(gradients[0][1][0], -0.123534462489121, 4)
        self.assertAlmostEqual(gradients[0][2][0], 0.360420387268185, 4)

    def test_calc_gradients_for_perceptron_with_dual_outputs(self):
        perceptron = MLNN([2, 2], act_func=sigmoid, eta=1)
        perceptron.weights = [ 
            np.array([
                [0.5, 0.5], 
                [0.1, 0.1], 
                [0.2, 0.2]
            ]) 
        ]

        X = np.array([
            [1, 1],
            [3, 2],
            [-1, 2]
        ])
        Y = np.array([
            [0, 1], 
            [1, 0], 
            [0, 1]
        ])

        ff_output = perceptron._feedforward(X)
        gradients = perceptron._calc_gradients(ff_output, Y)

        self.assertEqual(len(gradients), 1)
        self.assertCountEqual(gradients[0].shape, (3, 2))

        self.assertAlmostEqual(gradients[0][0][0], 0.25400630956647, 4)
        self.assertAlmostEqual(gradients[0][1][0], -0.123534462489121, 4)
        self.assertAlmostEqual(gradients[0][2][0], 0.360420387268185, 4)

        self.assertAlmostEqual(gradients[0][0][1], 0.004081357172687, 4)
        self.assertAlmostEqual(gradients[0][1][1], 0.410148859451296, 4)
        self.assertAlmostEqual(gradients[0][2][1], 0.074480179000913, 4)

    def test_calc_output_for_net_with_sigmoid_act_func(self):
        net = self.create_net211_with_sigmoid_act_func()

        X = np.array([
            [1, 1],
            [3, 2],
            [-1, 2]
        ])
        Y = np.array([[0], [1], [0]])

        yhat = net.predict(X)

        self.assertAlmostEqual(yhat[0][0], 0.769766346444634, 4)
        self.assertAlmostEqual(yhat[1][0], 0.773916123548593, 4)
        self.assertAlmostEqual(yhat[2][0], 0.769766346444634, 4)

    def test_calc_gradients_for_net_with_sigmoid_act_func(self):
        net = self.create_net211_with_sigmoid_act_func()

        X = np.array([
            [1, 1],
            [3, 2],
            [-1, 2]
        ])
        Y = np.array([[0], [1], [0]])

        ff_output = net._feedforward(X)
        gradients = net._calc_gradients(ff_output, Y)

        self.assertEqual(len(gradients), 2)

        self.assertEqual(gradients[1].shape, (2, 1))
        self.assertAlmostEqual(gradients[1][0][0], 0.233287516996703, 4)
        self.assertAlmostEqual(gradients[1][1][0], 0.157855149571173, 4)

        self.assertEqual(gradients[0].shape, (3, 1))
        self.assertAlmostEqual(gradients[0][0][0], 0.015398144806812472, 4)
        self.assertAlmostEqual(gradients[0][1][0], -0.006333415234832, 4)
        self.assertAlmostEqual(gradients[0][2][0], 0.022041648004414, 4)

    def test_calc_gradients_for_net_with_two_hidden_units(self):
        net = MLNN([2, 2, 1], act_func=sigmoid, eta=1)
        net.weights = [
            np.array([
                [0.5, 0.25],
                [0.1, 0.5],
                [0.2, 0.8]
            ]),
            np.array([
                [1], [0.3], [-0.1]
            ])
        ]

        X = np.array([
            [1, 1],
            [3, 2],
            [-1, 2]
        ])
        Y = np.array([[0], [1], [0]])

        ff_output = net._feedforward(X)
        gradients = net._calc_gradients(ff_output, Y)

        self.assertEqual(len(gradients), 2)

        self.assertEqual(gradients[1].shape, (3, 1))
        self.assertAlmostEqual(gradients[1][0][0], 0.234435076163075, 4)
        self.assertAlmostEqual(gradients[1][1][0], 0.158232494732687, 4)
        self.assertAlmostEqual(gradients[1][2][0], 0.182761780873289, 4)

        self.assertEqual(gradients[0].shape, (3, 2))

        self.assertAlmostEqual(gradients[0][0][0], 0.01552879333695, 4)
        self.assertAlmostEqual(gradients[0][1][0], -0.007170830669011, 4)
        self.assertAlmostEqual(gradients[0][2][0], 0.022093172309425, 4)

        self.assertAlmostEqual(gradients[0][0][1], -0.004152687151214, 4)
        self.assertAlmostEqual(gradients[0][1][1], 0.000704785568543, 4)
        self.assertAlmostEqual(gradients[0][2][1], -0.006287794062515, 4)