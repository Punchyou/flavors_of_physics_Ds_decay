Dataset Issues:
    * High acciracy due to the dataset
    * Pickec another dataset, but it was imbalanced
    * Had to make it balances by resampling the dataset


How to choose the best n components for pca:
* PCA project the data to less dimensions, so we need to scale the data
beforehand.

* A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. This can be determined by looking at the cumulative explained variance ratio as a function of the number of components
This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components.

* From the pca graph, looks like that the data are described from 3 variables.

Why use sklearn pipeline for SGDClassifier?
Often in ML tasks you need to perform sequence of different transformations (find set of features, generate new features, select only some good features) of raw dataset before applying final estimator. Pipeline gives you a single interface for all 3 steps of transformation and resulting estimator. It encapsulates transformers and predictors inside.


From he hist() plot of the training data:
If we ignore the clutter of the plots and focus on the histograms themselves, we can see that many variables have a skewed distribution.

The dataset provides a good candidate for using a robust scaler transform to standardize the data in the presence of skewed distributions and outliers.

How to do the capstonre project report: https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_report_template.md

TODO: The dataset has been resampled, due to the amount of the first dataset, only the resampled dataset is present in the project. The code used for resampling is the following:
```py
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

. The problem with accuracy this metric is that when problems are imbalanced it is easy to get a high accuracy score by simply classifying all observations as the majority class


### learning curves
"""The first plot is the learning curve
The plots in the second row show the times required by the models to train with various sizes of training dataset. The plots in the third row show how much time was required to train the models for each training sizes."""

Sources:
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
Data Exploration - Model selection:
https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
Data Science from Scratch
Machine learning with python
Feature Egnineering
https://github.com/Punchyou/blog-binary-classification-metrics
https://www.youtube.com/watch?v=aXpsCyXXMJE
Models Evaluation:
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    * https://neptune.ai/blog/evaluation-metrics-binary-classification
Hyperparameters Tuning:
    * https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
    * https://towardsdatascience.com/a-guide-to-svm-parameter-tuning-8bfe6b8a452c
Model Selection: https://machinelearningmastery.com/types-of-classification-in-machine-learning/
https://www.kaggle.com/nanomathias/bayesian-optimization-of-xgboost-lb-0-9769