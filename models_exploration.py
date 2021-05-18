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
from utils import gather_performance_metrics, range_inc
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold




def main():
    # load data
    # train_df = pd.read_csv('data/training.csv', index_col='id')
    train_df = pd.read_csv("data/resampled_data.csv", index_col="Unnamed: 0")
    y = train_df["signal"]
    X = train_df.drop("signal", axis=1)

    # split in X and y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42
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

    metrics_df = pd.DataFrame()
    for X_train, X_test, scale in [(X_train_minmax_scaled, X_test_minmax_scaled, "minmax"), (X_train_std_scaled, X_test_std_scaled, "std"), (X_train_rob_scaled, X_test_rob_scaled, "rob")]:

    
        # # model 1 - KNN
        knn_clf = KNeighborsClassifier()
        # make use of grid search as is relatively fast for knn
        knn_rs_cv = GridSearchCV(estimator=knn_clf, param_grid={'n_neighbors': range(1, 60)})
        knn_rs_cv.fit(X=X_train, y=y_train)
        knn_prediction = knn_rs_cv.predict(X_test)
        metrics_df = metrics_df.append(gather_performance_metrics(y_true=y_test,
                                                                  y_pred=knn_prediction,
                                                                  model_col=f"knn_{scale}"))
        # plotting prec/recall curve
        # plot_precision_recall_curve(estimator=knn_clf, X=X_train, y=y_train)
        # plt.show()
    
        # model 2 - SDG: Stohastic Gradient Descient (linear SVM)
        # minimizes the loss function and the regularization term (penalty, for overfitting)
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
                "alpha": list(range_inc(0.001, 50, 0.01, 1.5, 3)),
                "early_stopping": [True],
                "random_state": [42],
            },
        )
        sgd_rs_cv.fit(X_train, y_train)
        sgd_prediction = sgd_rs_cv.predict(X_test)
        metrics_df = metrics_df.append(gather_performance_metrics(y_test,
                                                                  sgd_prediction,
                                                                  f"sgd_{scale}"))
    
        # model 3 - support vector machines
        svm_clf = svm.SVC()
        svm_clf.fit(X_train, y_train)
        svm_prediction = svm_clf.predict(X_test)
    
        svm_rs_cv = RandomizedSearchCV(
            estimator=svm_clf,
            param_distributions={
                "C": list(range_inc(0.5, 100, 0.9, 1.2)),
                "break_ties": [False],
                "decision_function_shape": ["ovo", "ovr"],
                "kernel": ["poly", "rbf", "sigmoid"],
                "random_state": [42] 
            }
        )
        svm_rs_cv.fit(X_train, y_train)
        svm_prediction = svm_rs_cv.predict(X_test)
        metrics_df = metrics_df.append(gather_performance_metrics(y_true=y_test,
                                                                  y_pred=svm_prediction,
                                                                  model_col=f"svm_{scale}"))

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
    
        dtrain = xgb_dmatrix(X_train, label=y_train)
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
        xgb_bo.maximize(init_points=3, n_iter=5, acq='ei', random_state=42)
    
        bo_res = pd.DataFrame(xgb_bo.res)
        max_params = bo_res.loc[bo_res['target']==bo_res['target'].max(), "params"].values[0]
        max_params["max_depth"] = int(max_params["max_depth"])
        max_params["objective"] = "binary:logistic"
        max_params["random_state"]=42
        max_params_df = pd.DataFrame(max_params, index=['max_params'])
    
        # ------------ TESTING ---------------------
        # Train a new model with the best parameters from the search
        clf = XGBClassifier(**max_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics_df = metrics_df.append(gather_performance_metrics(y_test,
                                                                  y_pred,
                                                                  f"xgb_bayes_opt_{scale}"))
    
        # second way of optimizing + cv
        # Classifier
        bayes_cv_tuner = BayesSearchCV(
            estimator = XGBClassifier(
                n_jobs = 1,
                objective = 'binary:logistic',
                eval_metric = 'auc',
                learning_rate= 0.1,
                subsample=0.8,
                random_state=42
            ),
        search_spaces = {
                'learning_rate': (0.01, 1.0, 'log-uniform'),
            'max_depth': (3, 7),
                'min_child_weight': (1, 10),# no
                'gamma': (1e-9, 0.5, 'log-uniform'),#no
                'colsample_bytree': (0.01, 1.0, 'uniform'),#no
                'colsample_bylevel': (0.01, 1.0, 'uniform'),#no
                'reg_lambda': (1, 1000, 'log-uniform'),
                'reg_alpha': (1e-9, 1.0, 'log-uniform'),#no
                            'n_estimators': (50, 100),
                #'scale_pos_weight': (1e-6, 500, 'log-uniform')
    
    
                        },    
            scoring = 'accuracy',
            cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            ),
            n_iter = 10,
            verbose = 0,
            refit = True,
            random_state = 42
        )
        def status_print(optim_result):
            """Status callback durring bayesian hyperparameter search"""
            
            # Get all the models tested so far in DataFrame format
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
            
            # Get current parameters and the best parameters    
            best_params = pd.Series(bayes_cv_tuner.best_params_)
            print('Model #{}\nBest Accuracy: {}\nBest params: {}\n'.format(
                len(all_models),
                np.round(bayes_cv_tuner.best_score_, 4),
                bayes_cv_tuner.best_params_
            ))
            
            # Save all model results
            clf_name = bayes_cv_tuner.estimator.__class__.__name__
            all_models.to_csv(clf_name+"_cv_results.csv")
        result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
        y_pred = result.predict(X_test)
        metrics_df = metrics_df.append(gather_performance_metrics(y_test,
                                                                  y_pred,
                                                                  f"xgb_bayes_opt2_{scale}"))
        metrics_df.to_csv("metrics_results.csv")



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

