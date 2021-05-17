import matplotlib.pyplot as plt
import mypy
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import kstest
from sklearn import metrics, preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             plot_precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBClassifier, cv as xgb_cv, DMatrix as xgb_dmatrix, train as xgb_train
from bayes_opt import BayesianOptimization
from utils import *




def main():
    # TODO: remote * imports
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
    # X.hist(figsize=(60, 20))
    # plt.show()

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
    metrics_df = gather_performance_metrics(
        y_true=y_test, y_pred=knn_bsl_prediction, model_col="knn_baseline"
    )

    # # model 1 - KNN
    # knn_clf = KNeighborsClassifier()
    # # make use of grid search as is relatively fast for knn
    # knn_rs_cv = GridSearchCV(estimator=knn_clf, param_grid={'n_neighbors': range(1, 60)})
    # knn_rs_cv.fit(X=X_train_rob_scaled, y=y_train)
    # knn_prediction = knn_rs_cv.predict(X_test_rob_scaled)
    # metrics_df = metrics_df.append(gather_performance_metrics(y_true=y_test,
    #                                                           y_pred=knn_prediction,
    #                                                           model_col="knn"))
    # # plotting prec/recall curve
    # # plot_precision_recall_curve(estimator=knn_clf, X=X_train, y=y_train)
    # # plt.show()

    # # model 2 - SDG: Stohastic Gradient Descient (linear SVM)
    # # minimizes the loss function and the regularization term (penalty, for overfitting)
    # sgd_clf = SGDClassifier()
    # sgd_rs_cv = RandomizedSearchCV(
    #     estimator=sgd_clf,
    #     param_distributions={
    #         "loss": [
    #             "log",
    #             "hinge",
    #             "modified_huber",
    #             "squared_hinge",
    #             "perceptron",
    #             "squared_loss",
    #             "huber",
    #             "epsilon_insensitive",
    #             "squared_epsilon_insensitive",
    #         ],
    #         "penalty": ["l2", "l1"],
    #         "alpha": list(range_inc(0.001, 50, 0.01, 1.5, 3)),
    #         "early_stopping": [True],
    #         "random_state": [42],
    #     },
    # )
    # sgd_rs_cv.fit(X_train_rob_scaled, y_train)
    # sgd_prediction = sgd_rs_cv.predict(X_test_rob_scaled)
    # metrics_df = metrics_df.append(gather_performance_metrics(y_test,
    #                                                           sgd_prediction,
    #                                                           "sgd"))

    # # model 3 - support vector machines
    # svm_clf = svm.SVC()
    # svm_clf.fit(X_train_rob_scaled, y_train)
    # svm_prediction = svm_clf.predict(X_test_rob_scaled)

    # svm_rs_cv = RandomizedSearchCV(
    #     estimator=svm_clf,
    #     param_distributions={
    #         "C": list(range_inc(0.5, 100, 0.9, 1.2)),
    #         "break_ties": [False],
    #         "decision_function_shape": ["ovo", "ovr"],
    #         "kernel": ["poly", "rbf", "sigmoid"],
    #         "random_state": [42] 
    #     }
    # )
    # svm_rs_cv.fit(X_train_rob_scaled, y_train)
    # svm_prediction = svm_rs_cv.predict(X_test_rob_scaled)
    # metrics_df = metrics_df.append(gather_performance_metrics(y_true=y_test,
    #                                                           y_pred=svm_prediction,
    #                                                           model_col="svm"))

    # # metrics_df.to_csv('metrics_df_to_delete.csv')
    # clf1 = XGBClassifier()
    # clf1.fit(X_train_rob_scaled, y_train)
    # xgb_pred = clf1.predict(X_test_rob_scaled)
    # metrics_df = metrics_df.append(gather_performance_metrics(y_test,
    #                                                           xgb_pred,
    #                                                           "xgb"))

    # ---- add to a different script ----
        # model 4 - XGBoost Classifieri
        # params have to have the same number of values

    param_bounds={
        "n_estimators": (0, 1000),
        'max_depth': (3, 10),
        "learning_rate": (0.1, 0.8),
        "gamma": (0, 100),
        "min_child_weight": (1, 1000),
        "max_delta_step": (0, 10),
        "reg_alpha": (0, 3),
        "reg_lambda": (1, 4),
        }

    dtrain = xgb_dmatrix(X_train_rob_scaled, label=y_train)
    def xgb_evaluate(max_depth, gamma, colsample_bytree):
        params = {"objective": "binary:logistic",
                  'eval_metric': 'logloss',
                  "learning_rate": 0.1,
                  'max_depth': int(max_depth),
                  'subsample': 0.8,
                  'gamma': gamma,
                  'colsample_bytree': colsample_bytree,
                  "random_state": 42}
        # Used around 1000 boosting rounds in the full model
        cv_result = xgb_cv(params=params, dtrain=dtrain, nfold=5)

        # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
        return -1.0 * cv_result['test-logloss-mean'].iloc[-1]

    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                                 'gamma': (0, 1),
                                                 'colsample_bytree': (0.3, 0.8)
                                                 }, random_state=(42))
    xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')

    bo_res = pd.DataFrame(xgb_bo.res)
    max_params = bo_res.loc[bo_res['target']==bo_res['target'].max(), "params"].values[0]
    max_params["max_depth"] = int(max_params["max_depth"])
    max_params["objective"] = "binary:logistic"
    max_params["random_state"]=42
    max_params_df = pd.DataFrame(max_params, index=['max_params'])

    # ------------ TESTING ---------------------
    # Train a new model with the best parameters from the search
    clf = XGBClassifier(**max_params)
    clf.fit(X_train_rob_scaled, y_train)
    y_pred = clf.predict(X_test_rob_scaled)
    metrics_df = metrics_df.append(gather_performance_metrics(y_test,
                                                              y_pred,
                                                              "xgb_optimized5"))
    metrics_df.to_csv("results_to_delete.csv")

    # # Report testing and training RMSE
    # print(np.sqrt(mean_squared_error(y_test, y_pred)))
    # print(np.sqrt(mean_squared_error(y_train, y_train_pred)))


# TODO add viz where needed
    # sns.heatmap(
    #    confusion_matrix(y_test, y_pred), annot=[["tn", "fp"], ["fn", "tp"]], fmt="s"
    # )
    # plt.show()

    # check if the model suffers from bias as accuracy is very high
    # plot the learning curve
    # utils.plot_learning_curve(
    #     estimator=sgd_clf, X=X_train, y=y_train, n_jobs=8, title="title"
    # )
    # plt.show()

if __name__ == "__main__":
    main()

