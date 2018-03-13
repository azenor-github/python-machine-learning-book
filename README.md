## Python Machine Learning (2nd Ed.) - Progress Journal

Official repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition

### Table of Contents

1. - [x] Machine Learning - Giving Computers the Ability to Learn from Data
2. - [x] Training Machine Learning Algorithms for Classification
3. - [x] A Tour of Machine Learning Classifiers Using Scikit-Learn
4. - [x] Building Good Training Sets – Data Pre-Processing
5. - [x] Compressing Data via Dimensionality Reduction
6. - [x] Learning Best Practices for Model Evaluation and Hyperparameter Optimization
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

#### Chapter 05

1. What is the PCA? What are the main steps performed by PCA?
2. Read about eigenvalues (Linear Algebra and its applications).
3. What is the LDA? What are the differences between PCA & LDA?
4. What is the kernel PCA? When one should prefer kPCA over PCA? 

#### Chatper 06

1. What is a sckit-learn Pipeline? What are the basic methods of its interface?
2. What is the main idea behind holdout method? How do we split original dataset and what is the purpose of each created subset? What is the main disadventage of this method?
3. How does k-fold cross-validation method work? What is the optimal number of folds?
4. Read [Model evaluation, model selection, and algorithm selection in machine learning](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part1.html).
5. What are the learning and validation curves? How do we use them to diagnose the model?
6. What are the hyperparameters/metaparameters? How do we tune them? Hod do they differ from parameters learned from the training data (e.g. weights of logistic regression)?
7. Describe nested cross-validation method used for alogrithm selection.
8. What is a confusion matrix and how is it build?
9. How one can deal with inbalance sample?
10. (Titanic Kaggle Chellange) Select the best model for predicting what sort of people were likely to survive titanic catasthrope. You can create new features on the base of existing ones (features engineering).


### Aditional Resources

#### Chapter 04

1. [4.3. Preprocessing data](http://scikit-learn.org/stable/modules/preprocessing.html)
2. [Section - Should I normalize/standardize/rescale](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)
3. [About Feature Scaling and Normalization](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

#### Chatper 05

#####  Eigenvectors/Eigenvalues

1. [Understanding Eigenvectors and Eigenvalues Visually](https://alyssaq.github.io/2015/understanding-eigenvectors-and-eigenvalues-visually/)
2. [Linear Transforms and Eigenvectors](http://www.austinadee.com/wpblog/linear-transforms-and-eigenvectors/)

##### PCA

3. [PCA objective function: what is the connection between maximizing variance and minimizing error?](https://stats.stackexchange.com/questions/32174/pca-objective-function-what-is-the-connection-between-maximizing-variance-and-m/136072#136072)
4. [What is an intuitive explanation for how PCA turns from a geometric problem (with distances) to a linear algebra problem (with eigenvectors)?](https://stats.stackexchange.com/questions/217995/what-is-an-intuitive-explanation-for-how-pca-turns-from-a-geometric-problem-wit)
5. [Making sense of principal component analysis, eigenvectors & eigenvalues](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579#140579)

#### LDA

6. [LECTURE 10: Linear Discriminant Analysis](http://vision.eecs.ucf.edu/courses/cap5415/fall2011/Lecture-14.5-LDA.pdf)