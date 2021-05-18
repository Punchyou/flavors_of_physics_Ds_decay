#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:38:51 2021

@author: maria
"""

import matplotlib.pyplot as plt
import mypy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, cv as xgb_cv, DMatrix as xgb_dmatrix
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import RobustScaler
from utils import gather_performance_metrics
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

def main():
    # load data
    train_df = pd.read_csv("data/resampled_data.csv", index_col="Unnamed: 0")
    y = train_df["signal"]
    X = train_df.drop("signal", axis=1)

    # split in X and y train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Robust Scaler: doesn't take the median into account
    # only focuses on the parts where the bulk data is.
    rob_scaler = RobustScaler()
    X_train_rob_scaled = rob_scaler.fit_transform(X_train)
    X_test_rob_scaled = rob_scaler.fit_transform(X_test)

    # convert data to DMatrix, the xgboost's internal datastructure
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
    clf.fit(X_train_rob_scaled, y_train)
    y_pred = clf.predict(X_test_rob_scaled)
    gather_performance_metrics(y_test, y_pred, "xgb_optimized")

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
            'min_child_weight': (1, 10),
            'gamma': (1e-9, 0.5, 'log-uniform'),
            'colsample_bytree': (0.01, 1.0, 'uniform'),
            'colsample_bylevel': (0.01, 1.0, 'uniform'),
            'reg_lambda': (1, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'n_estimators': (50, 100),


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
    result = bayes_cv_tuner.fit(X_train_rob_scaled, y_train, callback=status_print)
    y_pred = result.predict(X_test_rob_scaled)
    gather_performance_metrics(y_test,
                                y_pred,
                                "xgb_bayes_opt2")


if __name__ == "__main__":
    main()