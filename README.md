## Python Machine Learning (2nd Ed.) - Progress Journal

Official repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition

### Table of Contents

1. - [x] Machine Learning - Giving Computers the Ability to Learn from Data
2. - [x] Training Machine Learning Algorithms for Classification
3. - [x] A Tour of Machine Learning Classifiers Using Scikit-Learn
4. - [x] Building Good Training Sets – Data Pre-Processing
5. - [ ] Compressing Data via Dimensionality Reduction
6. - [ ] Learning Best Practices for Model Evaluation and Hyperparameter Optimization
7. - [ ] Combining Different Models for Ensemble Learning
8. - [ ] Applying Machine Learning to Sentiment Analysis
9. - [ ] Embedding a Machine Learning Model into a Web Application
10. - [ ] Predicting Continuous Target Variables with Regression Analysis
11. - [ ] Working with Unlabeled Data – Clustering Analysis
12. - [ ] Implementing a Multi-layer Artificial Neural Network from Scratch
13. - [ ] Parallelizing Neural Network Training with TensorFlow
14. - [ ] Going Deeper: The Mechanics of TensorFlow
15. - [ ] Classifying Images with Deep Convolutional Neural Networks
16. - [ ] Modeling Sequential Data Using Recurrent Neural Networks


### Questions & Exercises

#### Chapter 02

1. Describe MCP-Neuron model.
2. Explain the perceptron learning rule.
3. Describe adaptive linear neuron model. What are the main differences to perceptron?
4. Describe gradient descent method and explain how it is used to optimize the weights in Adaline.
5. What are the differences between batch gradient descent and stochastic gradient descent?
6. Implement multi-class classifier that will enable to combine different kind of classifiers.
7. Implement mini-batch gradient descent.

#### Chapter 03

1. Describe logistic regression model.
2. Formulate gradient descent for logistic regression. Start from likelihood function.
3. Prctise sckit-learn for preprocessing dataset (splitting dataset, features scaling) on wine dataset.
4. Extend implementation of logistic regression model with by L2 regularization.
5. What is a SVM? What are the main advantages and drawbacks  of using SVM over logistic regression?
6. Read "Support Vector Machine" by Alexandre Kowalczyk (Sucinctly series).
7. Describe decision tree algorithm. What are the most typical impurity measures?
8. Implement decision tree classifier.
9. Read about bayesian learning from 'Machine Learning' by Tom Mitchel.
10. Describe Random Forest algorithm. What are the main metaparameters and how how they impact the overfitting or underfitting?
11. Describe KNN method. Why is it call 'lazy learner'?
12. Implment KNN.

#### Chapter 04

1. Read 'Intro to Data Structurs' from pandas doc (https://pandas.pydata.org/pandas-docs/stable/dsintro.html).
2. What are the typical strategies for dealing with missing values?
3. Read '4.3. Preprocessing data' from scikit-learn doc. (http://scikit-learn.org/stable/modules/preprocessing.html).
4. What are the typical methods for handling categorical data? What is one-hot encoding?
5. What are the most common feature scalling methods? Are then any reason to prefer any of them?
6. What is the purpose of regularization? What are the differences between L1 and L2 regularization methods?
7. Feature selection vs. feature extraction. Define both terms and explain the differences.
8. Implement forward feature selection algorithm.
9. How can one use random forests to assess importance of the features?
10. Read '1.13. Feature selection' from scikit-learn doc (http://scikit-learn.org/stable/modules/feature_selection.html).
11. Read 'Working with missing data' from pandas doc (https://pandas.pydata.org/pandas-docs/stable/missing_data.html).
12. Implment 'Exhaustive feature selection algorithm'. Evalute all possible combinations and choose the best subset.