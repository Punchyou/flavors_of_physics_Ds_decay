import logging
from pprint import pformat

import matplotlib.pyplot as plt
import mypy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from xgboost import XGBClassifier

from utils import (
    gather_performance_metrics,
    model_metrics_comparison_heatmap,
    plot_learning_curve,
)


class BayesSearchCVTuner:
    """
    A class that provides an abstractions for the the BayesSearchCV tuner.
    An XGBClassifier is tuned with Bayes optimization and 3-fold cross
    validation. The search parameters are:
        * objective: "binary:logistic"
        * learning_rate: (0.01, 1.0)
        * max_depth: (3, 7)
        * min_child_weight: (1, 10)
        * gamma: (1e-9, 0.5)
        * colsample_bytree: (0.01, 1.0)
        * colsample_bylevel: (0.01, 1.0)
        * reg_lambda: (1, 1000)
        * reg_alpha: (1e-9, 1.0)
        * n_estimators: (50, 100)
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.bayes_cv_tuner = BayesSearchCV(
            estimator=XGBClassifier(
                n_jobs=1,
                objective="binary:logistic",
                eval_metric="auc",
                # how many samples will xgboost randomly sample
                # before growing trees to prevent obverfitting
                subsample=0.8,
                use_label_encoder=False,
                random_state=42,
            ),
            search_spaces={
                "learning_rate": (0.01, 1.0),
                "max_depth": (3, 7),
                "min_child_weight": (1, 10),
                "gamma": (1e-9, 0.5),
                "colsample_bytree": (0.01, 1.0),
                "colsample_bylevel": (0.01, 1.0),
                "reg_lambda": (1, 1000),
                "reg_alpha": (1e-9, 1.0),
                "n_estimators": (50, 100),
            },
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_iter=10,
            refit=True,
            random_state=42,
        )

    def _status_print(self, optim_result) -> None:
        """
        Status callback durring bayesian hyperparameter search.

        Parameters
        ----------
        optim_result : scipy.optimize.optimize.OptimizeResult

        Returns
        -------
        None.
        """

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(self.bayes_cv_tuner.cv_results_)

        # Get current parameters and the best parameters
        best_params = pd.Series(self.bayes_cv_tuner.best_params_)
        logging.info(
            f"\nModel #{len(all_models)}\nBest AUC:{np.round(self.bayes_cv_tuner.best_score_, 4)}\nBest params:\n{pformat(dict(self.bayes_cv_tuner.best_params_), sort_dicts=False)}"
        )

        # Save all model results
        clf_name = self.bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(f"reports/{clf_name}_cv_results.csv", index=False)

    def fit(self):
        return self.bayes_cv_tuner.fit(
            self.X_train, self.y_train, callback=self._status_print
        )


def main():
    # load data
    train_df = pd.read_csv("data/resampled_data.csv")
    y = train_df["signal"]
    X = train_df.drop("signal", axis=1)

    # split in X and y train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # tune and fit
    bayes_tuner = BayesSearchCVTuner(X_train_scaled, y_train)
    result = bayes_tuner.fit()

    # predict
    y_pred = result.predict(X_test_scaled)
    metrics_df = gather_performance_metrics(
        y_test,
        y_pred,
        "xgb_bayes_tuned",
        best_params=dict(bayes_tuner.bayes_cv_tuner.best_params_),
    )
    logging.info(
        f"The model predicted signal events with accuracy {round(metrics_df['Accuracy'].values[0]* 100)}%"
    )

    # save metrics and prediction
    metrics_df.T.to_csv("reports/model_final_metrics.csv")
    prediction = y_test.to_frame()
    prediction["y_pred"] = y_pred
    prediction.to_csv("reports/signal_final_prediction.csv")

    # check if the model suffers from bias
    # plot the learning curve
    plot_learning_curve(
        estimator=result.best_estimator_,
        X=X_train_scaled,
        y=y_train,
        n_jobs=8,
        title="XGBoost Classifier",
    )
    plt.savefig("images/learning_curve.png", bbox_inches="tight", pad_inches=0.01)

    # comparison with the benchmark model
    knn = KNeighborsClassifier(n_neighbors=52)
    knn.fit(X_train, y_train.values)
    y_pred = knn.predict(X_test)
    metrics_df = metrics_df.append(
        gather_performance_metrics(y_test, y_pred, "knn_benchmark", best_params="k=52")
    )
    model_metrics_comparison_heatmap(df=metrics_df.drop("Best Parameters", axis=1))
    plt.savefig("images/comparison_heatmap.png", bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%d-%b%y %H: %M: %S",
        level=logging.INFO,
    )
    main()
