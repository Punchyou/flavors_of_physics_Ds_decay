import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve
import pandas as pd
import mypy
from scipy.stats import kstest
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, f1_score,
                             plot_precision_recall_curve, precision_score,
                             recall_score)


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


def range_inc(start: float, stop: float, step: float, inc: float=1, dec_pl: int=2):
    while start < stop:
        yield round(start, dec_pl)
        start += step
        step += start * inc


def get_column(df, col_num):
    # jth element of row A_i for eadch row A_i
    return df.columns[col_num]


def subplot_correlation_matrix(df, subplots_figsize=(30, 30)):
    # check  correlation matrix
    _, num_columns = np.shape(df)
    fig, ax = plt.subplots(num_columns, num_columns, figsize=subplots_figsize)
    for i in range(num_columns):
        for j in range(num_columns):
            if i != j:
                ax[i][j].scatter(get_column(df, j), get_column(df, i))
            else:
                ax[i][j].annotate("series" + str(i), (0.5, 0.5),
                                  xycoords='axes fraction',
                                  ha='center',
                                  va='center')
            if i < num_columns - 1:
                ax[i][j].xaxis.set_visible(False)
            if j > 0:
                ax[i][j].yaxis.set_visible(False)
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())
    plt.show()


# TODO change docstring style
def plot_heatmap(df, columns, vmin=-1, vmax=1, cmap="RdBu", annot=True,
                 figsize=(30, 30), title='Heatmap', title_fontsize=18, xticks_fontsize=14,
                 yticks_fontsize=14, annot_fontsize=8):
    """
    Parameters:
    -----------
    df      : dataframe source of the data                  : dataframe : :
    columns : list of the columns to be included            : str       : :
    dim     : tuple of the dimensions of the graph          : int       : :
    title   : title of the graph                            : str       : :
    vmin    : minimum correlation value                     : int       : :
    vmax    : maximum correlation value                     : int       : :
    cmap    : the color scheme to be used                   : str       : :
    annot   : whether or not the heat map will be annotated : Bool      : :

    Description:
    ------------
    Plots a heatmap for columns containing continuous data in a Pandas dataframe and allows for increased appearance control.
    The resulting heatmap is not mirrored. Skips 0 correlations.

    Returns:
    --------
    A heat map displaying the correlations between n number of columns.
    """
    plt.figure(figsize=figsize, facecolor="white")
    plt.title(f"{title}", size=title_fontsize)
    corr = df[columns].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, cmap=cmap,  mask=mask, vmin=vmin, vmax=vmax,
                    annot=annot, annot_kws={"fontsize": annot_fontsize})
    plt.xticks(size=xticks_fontsize)
    plt.yticks(size=yticks_fontsize)


# TODO check doscstring + add params
def display_component(pca_fitted_model, num_of_components, features_list, component_number, n_weights_to_display=10):
    """
    Examine the makeup of each PCA component based on the weightings
    of the original features that are included in the component.
    Note that the components are ordered from smallest to largest
    and so we are getting the correct rows by calling
    num_of_components-component_number to get the component_number
    component.
    """
    # get index of component (last row - component_num)
    row_idx = num_of_components - component_number

    components_makeup = pd.DataFrame(pca_fitted_model.components_)
    # get the list of weights from a row in v, dataframe
    v_1_row = components_makeup.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe
    comps = pd.DataFrame(list(zip(v_1, features_list)),
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights'] = comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights',
                                           ascending=False).head(n_weights_to_display)

    # display using seaborn
    ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(data=sorted_weight_data,
                     x="weights",
                     y="features",
                     palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_number))
    plt.show()


# TODO add docstring
def create_transformed_df(pca_components, scaled_train_df, num_of_components, n_top_components):
    # create new dataframe to add data to
    df = pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in pca_components:
        # get component values for each data point
        components = data.label['projection'].float32_tensor.values
        df = df.append([list(components)])

    # index by county, just like counties_scaled
    df.index = scaled_train_df.index

    # keep only the top n components
    start_idx = num_of_components - n_top_components
    df = df.iloc[:, start_idx:]

    # reverse columns, component order
    return df.iloc[:, ::-1]


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def scatterplot_range_knn_score(scores: list, from_: int=0, to=10, figsize:
                                   tuple= (10, 6)):
    """
    Plot accuracy or error score of a knn classifier.
    """
    plt.figure(figsize=figsize)
    plt.plot(
        range(from_, to),
        scores,
        color='blue',
        linestyle='dashed',
        marker='o',
        markerfacecolor='red',
        markersize=10
    )
    plt.title('Accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-", max(scores), "at K =", scores.index(max(scores)))
    return plt


