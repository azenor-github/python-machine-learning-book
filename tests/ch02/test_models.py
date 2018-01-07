import unittest
from unittest import mock
import operator

import numpy as np

import ch2
from ch2.models import *


class PerceptronTest(unittest.TestCase):
    
    def test_init_weights_initializes_weights(self):
        per = Perceptron()

        self.assertIsNone(per.weights)
        per.init_weights(5)
        self.assertIsNotNone(per.weights)

    def test_init_weights_for_one_seed_creates_identical_weights(self):
        weights_ref = Perceptron().init_weights(4, random_state=1).weights
        weights_wal = Perceptron().init_weights(4, random_state=1).weights

        self.assertTrue(np.array_equal(weights_ref, weights_wal))

    def test_init_weights_for_different_seeds_creates_different_weights(self):
        weights_ref = Perceptron().init_weights(4, random_state=1).weights
        weights_wal = Perceptron().init_weights(4, random_state=2).weights

        self.assertFalse(np.array_equal(weights_ref, weights_wal))

    def test_net_input_returns_dot_product_of_weights_and_inputs(self):
        per = Perceptron()
        per.w_ = np.array([1, 1, 1, 0])

        xi = np.array([1, 2, 3])
        z = per.net_input(xi)

        self.assertEqual(z, 4)

    def test_predict_predicts_positive_class_when_net_input_greater_than_zero(self):
        per = Perceptron()
        per.w_ = np.array([1, 1, 1, 0])

        output = per.predict(np.array([1, 2, 3]))

        self.assertEqual(output, 1)

    def test_predict_preditcs_negative_class_when_net_input_smaller_than_zero(self):
        per = Perceptron()
        per.w_ = np.array([1, 1, 1, 0])

        output = per.predict(np.array([-1, -2, -3]))

        self.assertEqual(output, -1)

    def test_update_weights_updates_weights_in_accordance_with_perceptron_learning_rule(self):
        per = Perceptron(eta=1)
        per.w_ = np.array([1, 1, 1, 0])

        output = np.array([-1]).reshape(-1, 1)
        xi = np.array([1, 2, 3]).reshape(1, -1) # yhat = 1

        errors = per.update_weights(xi, output)

        self.assertEqual(per.weights[0], -1)
        self.assertEqual(per.weights[1], -1)
        self.assertEqual(per.weights[2], -3)
        self.assertEqual(per.weights[3], -6)
        self.assertEqual(errors, 1)


class AdalineGDTest(unittest.TestCase):

    @mock.patch.object(ch2.models.AdalineGD, "init_weights")
    def test_fit_adaptive_linear_neuron_model(self, iw_mock):
        model = AdalineGD(eta=1, n_iter=1)
        model.w_ = np.array([1, 1, 1])

        y = np.array([-1, 1])
        X = np.array([[1, 1], [1, 2]])

        model.fit(X, y)

        self.assertTrue(np.array_equal(model.weights, np.array([-6, -6, -9])))

    def test_net_input_returns_dot_product_of_weights_and_inputs(self):
        per = AdalineGD()
        per.w_ = np.array([1, 1, 1, 0])

        xi = np.array([1, 2, 3])
        z = per.net_input(xi)

        self.assertEqual(z, 4)

    def test_update_weights_updates_weights_in_accordance_with_gradient_descent(self):
        model = AdalineGD(eta=1, n_iter=1)
        model.w_ = np.array([1, 1, 1])

        y = np.array([-1, 1])
        X = np.array([[1, 1], [1, 2]])

        model.update_weights(X, y)

        self.assertTrue(np.array_equal(model.weights, np.array([-6, -6, -9])))

    def test_activation_returns_identity(self):
        model = AdalineGD(eta=1, n_iter=1)

        value = model.activation(np.array([1, 2, 3]))

        self.assertTrue(np.array_equal(value, np.array([1, 2, 3])))



class MultiClassClassifierTest(unittest.TestCase):

    def test_creates_OvA_samples_with_respect_to_target(self):
        y = np.array([-1, 0, 1, -1, 0, 1])
        X = np.array([[-1, -1], [0, 0], [1, 1], [-1, -1], [0, 0], [1, 1]])

        mclf = MultiClassClassifier(clf=mock.Mock())

        samples = list(mclf.create_train_samples(X, y))

        self.assertEqual(len(samples), 3)

        sample = next(x for x in samples if x["cls"] == -1)

        self.assertTrue(np.array_equal(sample["y"], np.array([-1, 1, 1, -1, 1, 1])))
        self.assertTrue(np.array_equal(sample["X"], X))

    def test_fit_initializes_classifiers(self):
        clfs = [ mock.Mock(), mock.Mock(), mock.Mock() ]
        clf_cls = mock.Mock()
        clf_cls.side_effect = clfs

        mclf = MultiClassClassifier(clf=clf_cls, eta=1)

        y = np.array([-1, 0, 1, -1, 0, 1])
        X = np.array([[-1, -1], [0, 0], [1, 1], [-1, -1], [0, 0], [1, 1]])
        mclf.fit(X, y)

        self.assertEqual(clf_cls.call_count, 3)

        self.assertIn("eta", clf_cls.call_args_list[0][1])
        self.assertEqual(clf_cls.call_args_list[0][1]["eta"], 1)

    def test_fit_passess_proper_samples_to_classifiers(self):
        clfs = [ mock.Mock(), mock.Mock(), mock.Mock() ]
        clf_cls = mock.Mock()
        clf_cls.side_effect = clfs

        mclf = MultiClassClassifier(clf=clf_cls, eta=1)

        y = np.array([-1, 0, 1, -1, 0, 1])
        X = np.array([[-1, -1], [0, 0], [1, 1], [-1, -1], [0, 0], [1, 1]])
        mclf.fit(X, y)

        self.assertTrue(np.array_equal(clfs[0].fit.call_args[0][0], X))
        self.assertTrue(np.array_equal(
            clfs[0].fit.call_args[0][1], 
            np.array([-1, 1, 1, -1, 1, 1])
        ))

    def test_predict_calls_specific_method_of_classifiers(self):
        clfs = [ mock.Mock(), mock.Mock(), mock.Mock() ]
        clfs[0].fit.return_value = clfs[0]
        clfs[1].fit.return_value = clfs[1]
        clfs[2].fit.return_value = clfs[2]
        clfs[0].test.return_value = np.array([0])
        clfs[1].test.return_value = np.array([1])
        clfs[2].test.return_value = np.array([2])
        clf_cls = mock.Mock()
        clf_cls.side_effect = clfs

        mclf = MultiClassClassifier(clf=clf_cls, eta=1)

        y = np.array([-1, 0, 1, -1, 0, 1])
        X = np.array([[-1, -1], [0, 0], [1, 1], [-1, -1], [0, 0], [1, 1]])
        mclf.fit(X, y)

        y = mclf.predict(np.array([[1, 1]]), method=operator.attrgetter("test"))

        self.assertTrue(clfs[0].test.call_count, 1)
        self.assertTrue(clfs[1].test.call_count, 1)
        self.assertTrue(clfs[2].test.call_count, 1)

    def test_predict_predicts_classes_with_max_values(self):
        clfs = [ mock.Mock(), mock.Mock(), mock.Mock() ]
        clfs[0].fit.return_value = clfs[0]
        clfs[1].fit.return_value = clfs[1]
        clfs[2].fit.return_value = clfs[2]
        clfs[0].test.return_value = np.array([0])
        clfs[1].test.return_value = np.array([1])
        clfs[2].test.return_value = np.array([2])
        clf_cls = mock.Mock()
        clf_cls.side_effect = clfs

        mclf = MultiClassClassifier(clf=clf_cls, eta=1)

        y = np.array([-1, 0, 1, -1, 0, 1])
        X = np.array([[-1, -1], [0, 0], [1, 1], [-1, -1], [0, 0], [1, 1]])
        mclf.fit(X, y)

        y = mclf.predict(np.array([[1, 1]]), method=operator.attrgetter("test"))

        self.assertEqual(y[0], 1)


class AdalineMBGDTest(unittest.TestCase):

    def test_split_sample_into_batches(self):
        model = AdalineMBGD(batch=2)

        y = np.array([1, 0, 1, 0])
        X = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]])

        batches = model._split_into_batches(X, y)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0].shape[0], 2)
        self.assertEqual(batches[1][0].shape[0], 2)

    def test_create_batches_when_min_batch_size_greater_than_n(self):
        model = AdalineMBGD(batch=100)

        y = np.array([1, 0, 1, 0])
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

        batches = model._split_into_batches(X, y)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], 4)

    def test_create_batches_when_not_enough_n_to_fill_all_batches(self):
        model = AdalineMBGD(batch=3)

        y = np.array([1, 0, 1, 0])
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

        batches = model._split_into_batches(X, y)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0].shape[0], 3)
        self.assertEqual(batches[1][0].shape[0], 1)

    def test_shuffle_sample(self):
        model = AdalineMBGD(batch=3, random_state=1)

        y = np.array([1, 0, 1, 0])
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

        X2, y2 = model._shuffle(X, y)

        self.assertFalse(np.array_equal(X, X2))
        self.assertFalse(np.array_equal(y, y2))

    @mock.patch.object(ch2.models.AdalineMBGD, "init_weights")
    def test_fit_adaline_using_mini_batch_gradient_descent(self, iw_mock):
        model = AdalineMBGD(eta=1, n_iter=1, batch=2, shuffle=False)
        model.w_ = np.array([1, 1, 1])

        y = np.array([-1, 1, -1, 1])
        X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

        model.fit(X, y)

        self.assertTrue(np.array_equal(model.weights, np.array([7, -4, 3])))