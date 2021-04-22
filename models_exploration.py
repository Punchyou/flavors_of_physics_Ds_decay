import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef, plot_precision_recall_curve
from sklearn.pipeline import make_pipeline
import sklearn as sk
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import mypy

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


def gather_performance_metrics(y_true: list, y_pred: list, model_col: str) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # false negative rate
    fnr = false_negative_rate(tp, fn)
    npv = negative_predictive_value(tn, fn)
    fpr = false_positive_rate(fp, tn)
    # true positive rate - how many observations out of all positive
    # observations have we classified as positive
    # we want this to be close to 1
    recall = recall_score(y_true, y_pred)

    # positive predictive value - how many observations predicted as positive are in fact positive.
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # accuracy - how many observations, both positive and negative, were correctly
    # classified. The problem with this metric is that when problems are imbalanced it is easy to get a high accuracy score by simply classifying all observations as the majority class
    accuracy = accuracy_score(y_test, y_pred)
    return pd.DataFrame(
        data=[accuracy, fnr, npv, fpr, recall, precision, f1],
        index=[
            'Accuracy',
            'FNR',
            'NPV',
            'FPR',
            'Recall',
            'Precision',
            'F1'
        ],
        columns=[model_col]
    )


def main():
    # TODO add pipeline for models that take in scaled data
    # TODO add a func to fit and return predictions
    # TODO make clf var names consistent
    # load data
    train_df = pd.read_csv('data/training.csv', index_col='id')
    y = train_df['signal']
    X = train_df.drop('signal', axis=1)

    # split in X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # scaling is required for some of the following models
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # model 1 - KNN
    # find optimal value based on error
    from_, to = 1, 40
    error_rate = pd.DataFrame(index=range(from_, to), columns=['rate'])
    for i in range(from_, to):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_scaled, y_train)
        pred_i = knn.predict(X_test)
        error_rate.loc[error_rate.index==i] = np.mean(pred_i != y_test)

    # find optimal k values based on accuracy
    acc = []
    # Will take some time
    from_, to = 1, 10
    for i in range(from_, to):
        neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
        yhat = neigh.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, yhat))

    # apply knn with the oprimal k value
    k = 72
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train_scaled, y_train)
    knn_prediction = knn_clf.predict(X_test_scaled)
    # plotting prec/recall curve
    plot_precision_recall_curve(estimator=clf, X=X_train, y=y_train)
    plt.show()

    # model 2 - SDG - linear SVM
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    sgd_prediction = sgd_clf.predict(X_test)

    # Measure performance with predictions of CV and confusion matrix
    # might need to scale the results - easier with a pipeline, you get the
    # predictions right away
    sgd_pipeline = make_pipeline(StandardScaler(), SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    # return PREDICTIONS after CV - can only be used for prediction from training
    # data
    sgd_prediction_train = cross_val_predict(sgd_pipeline, X_train, y_train)

    # check how many times the clf did the job correctly
    cross_val_score(sgd_pipeline, X_train, y_train)

    # model 3 - support vector machines
    svm = svm.LinearSVC()
    svm.fit(X_train_scaled, y_train)
    svm_prediction = svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_prediction)
    # use cross validation as results acc is 1
    cross_val_score()

    # model 4 - Random forest - reduce overfitting
    rfc = RandomForestClassifier(random_state      = 42,
                                 n_estimators      = 6,
                                 max_depth         = 3,
                                 min_samples_split = 3,
                                 min_samples_leaf  = 2)

    rfc.fit(X_train, y_train)
    rfc_prediction = rfc.predict(X_test)
    accuracy_score(y_tes)

    # check if the model suffers from bias as accuracy is very high
    # plot the learning curve
    plot_learning_curve(estimator=rfc, X=X_train, y=y_train, n_jobs=8, title='title')
    plt.show()

    # model 5 - XGBoost Classifier
    cross_val_score(XGBClassifier(), X, y)

    # TODO add viz where needed
    # viz confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=[['tn', 'fp'],['fn', 'tp']], fmt='s')
    plt.show()

"""
How to choose the best n components for pca:
* PCA project the data to less dimensions, so we need to scale the data
beforehand.

* A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. This can be determined by looking at the cumulative explained variance ratio as a function of the number of components
This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components.

* From the pca graph, looks like that the data are described from 3 variables.

Why use sklearn pipeline for SGDClassifier?
Often in ML tasks you need to perform sequence of different transformations (find set of features, generate new features, select only some good features) of raw dataset before applying final estimator. Pipeline gives you a single interface for all 3 steps of transformation and resulting estimator. It encapsulates transformers and predictors inside.

Sources:
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
Data Science from Scratch
Machine learning with python
Feature Egnineering
https://github.com/Punchyou/blog-binary-classification-metrics
https://www.youtube.com/watch?v=aXpsCyXXMJE
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

"""
