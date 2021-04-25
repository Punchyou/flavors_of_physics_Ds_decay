import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
import mypy
from scipy.stats import kstest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    plot_precision_recall_curve,
)

# performance metrics funcs
def false_negative_rate(tp: float, fn: float) -> float:
    # flase negative rate, type 2 error - When we don’t predict something when it is, we are contributing to the false negative rate
    # we want this to be close to 0
    return fn / (tp + fn)


def negative_predictive_value(tn: float, fn: float) -> float:
    # Negative Predictive Value - measures how many predictions out of all negative
    # predictions were correct
    # we want this to be close to 1
    return tn / (tn + fn)


def false_positive_rate(fp: float, tn: float) -> float:
    # false positive rate, type 1 error - When we predict something when it isn’t we are contributing to the false positive rate
    # we want this to be close to 0
    return fp / (fp + tn)


def true_negative_rate(tn: float, fp: float) -> float:
    return tn / (tn + fp)


def error_score(y_true: list, y_pred: list):
    return np.mean(y_pred != y_true)


def gather_performance_metrics(
    y_true: list, y_pred: list, model_col: str
) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # false negative rate
    fnr = false_negative_rate(tp, fn)
    npv = negative_predictive_value(tn, fn)
    fpr = false_positive_rate(fp, tn)
    tnr = true_negative_rate(tn, fp)
    # true positive rate or sensitivity - how many observations out of all positive
    # observations have we classified as positive
    # we want this to be close to 1
    recall = recall_score(y_true, y_pred)

    # positive predictive value - how many observations predicted as positive are in fact positive.
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # accuracy - how many observations, both positive and negative, were correctly
    # classified. The problem with this metric is that when problems are imbalanced it is easy to get a high accuracy score by simply classifying all observations as the majority class
    accuracy = accuracy_score(y_true, y_pred)
    error = error_score(y_true, y_pred)
    ks = kstest(y_pred, y_true)[0]
    return pd.DataFrame(
        data=[[accuracy, error, ks, fnr, npv, fpr, tnr, recall, precision, f1]],
        columns=[
            "Accuracy",
            "Error",
            "KS",
            "FNR",
            "NPV",
            "FPR",
            "TNR",
            "Recall",
            "Precision",
            "F1",
        ],
        index=[model_col],
    )


def range_inc(start, stop, step, inc):
    i = start
    while i < stop:
        yield i
        i += step
        step += i * inc


def main():
    # TODO add pipeline for models that take in scaled data
    # TODO add a func to fit and return predictions
    # TODO add a baseline model
    # load data
    # train_df = pd.read_csv('data/training.csv', index_col='id')
    train_df = pd.read_csv("check_agreement.csv", index_col="id")
    y = train_df["signal"]
    # according to the dataset description, weights are used to
    # determine if a decay is signal or background event
    X = train_df.drop(["weight", "signal"], axis=1)

    # TODO move it to the data exporation file
    # check distributions
    X.hist(figsize=(60, 20))
    plt.show()

    # undersample to create balanced dataset
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # split in X and y
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2
    )

    # TODO move to the data exploration file
    # scaling is required for some of the following models
    minmax_scaler = MinMaxScaler()
    X_train_minmax_scaled = minmax_scaler.fit_transform(X_train)
    X_test_minmax_scaled = minmax_scaler.fit_transform(X_test)

    # TODO move to data expl file
    std_scaler = StandardScaler()
    X_train_std_scaled = std_scaler.fit_transform(X_train)
    X_test_std_scaled = std_scaler.fit_transform(X_test)

    # Robust Scaler: doesn't take the median into account and only focuses on the parts where the bulk data is.
    rob_scaler = RobustScaler()
    X_train_rob_scaled = rob_scaler.fit_transform(X_train)
    X_test_rob_scaled = rob_scaler.fit_transform(X_test)

    # baseline model
    knn_bsl = KNeighborsClassifier()
    knn_bsl.fit(X_train_rob_scaled, y_train)
    knn_bsl_prediction = knn_bsl.predict(X_test_rob_scaled)
    gather_performance_metrics(
        y_true=y_test, y_pred=knn_bsl_prediction, model_col="knn_baseline"
    )
    # model 1 - KNN
    # parameter tuning
    from_ = 1
    to = 60
    for i in range(from_, to):
        model_col = f"k={i}"
        knn_clf = KNeighborsClassifier(n_neighbors=i)
        knn_clf.fit(X_train_rob_scaled, y_train)
        knn_prediction = knn_clf.predict(X_test_rob_scaled)
        # need to transfotm returned df into a pd.Series to match the columns
        if i != from_:
            knn_metrics_rate.loc[model_col] = gather_performance_metrics(
                y_true=y_test, y_pred=knn_prediction, model_col=model_col
            ).squeeze()
        else:
            knn_metrics_rate = gather_performance_metrics(
                y_true=y_test, y_pred=knn_prediction, model_col=model_col
            )
    knn_max_acc = knn_metrics_rate["Accuracy"].max()
    best_k = knn_metrics_rate[knn_metrics_rate["Accuracy"] == knn_max_acc].index

    # plotting prec/recall curve
    plot_precision_recall_curve(estimator=knn_clf, X=X_train, y=y_train)
    plt.show()

    # model 2 - SDG: Stohastic Gradient Descient (linear SVM)
    # minimizes the loss function and the regularization term (penalty, for
    # overfitting)
    sgd_clf = SGDClassifier()
    sgd_rs_cv = RandomizedSearchCV(
        estimator=sgd_clf,
        param_distributions={
            "loss": [
                "log",
                "hinge",
                "modified_huber",
                "squared_hinge",
                "perceptron",
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
            "penalty": ["l2", "l1"],
            "alpha": [round(i, 5) for i in range_inc(0.0001, 100, 0.0001, 1.05)],
            "early_stopping": [True, False],
            "random_state": [42],
        },
    )
    sgd_rs_cv.fit(X_train_rob_scaled, y_train)
    sgd_prediction = sgd_rs_cv.predict(X_test_rob_scaled)
    gather_performance_metrics(y_test, sgd_prediction, "sgd")
    sgd_rs_cv.best_params_

    # model 3 - support vector machines
    svm_clf = svm.SVC()
    svm_clf.fit(X_train_rob_scaled, y_train)
    svm_prediction = svm_clf.predict(X_test_rob_scaled)

    svm_rs_cv = RandomizedSearchCV(
        estimator=svm_clf,
        param_distributions={
            "C": [round(i, 5) for i in range_inc(0.5, 500, 0.5, 1.2)],
            "break_ties": [True],
            "decision_function_shape": ["ovo", "ovr"],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
            "degree": [i for i in range(2, 10)],
        },
    )
    svm_rs_cv.fit(X_train_rob_scaled, y_train)

    # model 4 - XGBoost Classifier
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    xgb_prediction = xgb_clf.predict(X_test)

    # TODO add viz where needed
    sns.heatmap(
        confusion_matrix(y_test, y_pred), annot=[["tn", "fp"], ["fn", "tp"]], fmt="s"
    )
    plt.show()

    # check if the model suffers from bias as accuracy is very high
    # plot the learning curve
    utils.plot_learning_curve(
        estimator=sgd_clf, X=X_train, y=y_train, n_jobs=8, title="title"
    )
    plt.show()


"""
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
"""
