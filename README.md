## Python Machine Learning (2nd Ed.) - Progress Journal

Official repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition

### Table of Contents

1. - [x] Machine Learning - Giving Computers the Ability to Learn from Data
2. - [x] Training Machine Learning Algorithms for Classification
3. - [x] A Tour of Machine Learning Classifiers Using Scikit-Learn
4. - [x] Building Good Training Sets – Data Pre-Processing
5. - [x] Compressing Data via Dimensionality Reduction
6. - [x] Learning Best Practices for Model Evaluation and Hyperparameter Optimization
7. - [x] Combining Different Models for Ensemble Learning
8. - [x] Applying Machine Learning to Sentiment Analysis
9. - [x] Embedding a Machine Learning Model into a Web Application
10. - [x] Predicting Continuous Target Variables with Regression Analysis
11. - [x] Working with Unlabeled Data – Clustering Analysis
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

#### Chapter 07

1. What is the main idea behind ensemble methods? How does they make predictions?
2. Stacking vs. majority voting? What is the difference?
3. How do bagging and boosting work? What are the benefits of using them and what problems thay can solve?
4. Describe AdaBoost method and its learning procedure.
5. Read [1.11. Ensemble methods](http://scikit-learn.org/stable/modules/ensemble.html).
6. Learn about gradient boosting including XGBoost. Do some example using wine dataset.
7. (Titanic Kaggle Chellange) Use ensemble methods to improve accuracy of your predictions.

#### Chapter 08

1. What is the sentimental analysis?
2. What is the main idea behind bag-of-words representation?
3. Calculate tf-idf of a word 'sun' in the first document (p. 259).
4. What are the typical steps performing during text cleaning?
5. Read aboug Naive Bayes Classifier.
6.* Read about word2vec algorithm.
7. What is the purpose of topic modelling ?
9. Read "Latent Dirichlet Allocation towords a deeper understanding".
10. Build logistic regression model for classifying spam SMS (Kaggle).

#### Chatper 09

1. Read more about RANSAC algorithm.
2. Use regularized methods (Ridge Regression, LASSO) for predicting housing 
prices and compare the performance of the models on independed sample.
3. Create regression model for predicting students' final grade (kaggel dataset).
4. What is a quantile regression? When one should prefere it over ordinal 
linear regression?

#### Chapter 10

1. What is a clustering? How does unsupervised learning differ from suppervised learning?
2. What are the three types of clustering methods? Name one algorithm from each type.
3. Describe k-means algorithm. What is the main disadventage of k-means?
4. Hard vs. soft clustering? What are the differences?
5. How do we assess the quality of clustering? How one can choose the number of clusterings?
6. What is the main idea behind hierarchical clustering? What is the difference between agglomerative and divisive clustering? Name the most important parameters affecting the creation of clusters.
7. What is DBSCAN?
8. Read about spectral clustering. [A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189.pdf)
9. Apply clustering methods to group handwritten digits.

### Aditional Resources

[Artificial Intelligence  MIT Course](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/)

#### Chapter 03

1. [Random Forest From Top To Bottom](https://gormanalysis.com/random-forest-from-top-to-bottom/)

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

##### LDA
6. [LECTURE 10: Linear Discriminant Analysis](http://vision.eecs.ucf.edu/courses/cap5415/fall2011/Lecture-14.5-LDA.pdf)

#### Chapter 07

1. [CatBoost vs. Light GBM vs. XGBoost](https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html)
2. [Gradient Boosting from scratch](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
3. [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
4. [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
5. [17. Learning: Boosting](https://www.youtube.com/watch?v=UHBmv7qCey4)
6. [Trevor Hastie - Gradient Boosting Machine Learning](https://www.youtube.com/watch?v=wPqtzj5VZus)
7. [Bagging predictors, L.Breiman, Machine learning](https://www.stat.berkeley.edu/~breiman/bagging.pdf)
8. [The Strength of Weak Learnability, R.E. Schapire, Machine Learning](http://rob.schapire.net/papers/strengthofweak.pdf)
9. [Gradient Boosted Feature Selection](http://alicezheng.org/papers/gbfs.pdf)
10. [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/model.html)
11. [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
12. [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)


#### Chapter 08

1. [Naive Bayes and Text Classification I Introduction and Theory](https://arxiv.org/pdf/1410.5329v3.pdf)
2. [Word2vec Tutorial](https://rare-technologies.com/word2vec-tutorial/#word2vec_tutorial)
3. [Lecture 2 | Word Vector Representations: word2vec](https://www.youtube.com/watch?v=ERibwqs9p38)
4. [EM algorithm: how it works](https://www.youtube.com/watch?v=REypj2sy_5U)
5. [Latent Dirichlet Allocation: Towards a Deeper Understanding](http://obphio.us/pdfs/lda_tutorial.pdf)


#### Chatper 10

1. [Robust Estimation : RANSAC](http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf)
2. [Random sample consensus - Wikipedia](https://en.wikipedia.org/wiki/Random_sample_consensus)