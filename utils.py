import matplotlib.pyplot as plt
import mypy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import learning_curve

"""
This script contains unitility functions that has been used for this project.
It is a collection of metrics and visulization functions.
"""

# performance metrics funcs
def false_negative_rate(tp: float, fn: float) -> float:
    """
    False negative rate (type 2 error)
    When we don’t predict something when it is,
    we are contributing to the false negative rate
    We want this to be close to 0

    Calculation:
        false_negative / (true positives + false negatives)

    Parameters
    ----------
    tp : float
        number of true positives
    fn : float
        number of false negatives.

    Returns
    -------
    float
        false negative rate.

    """
    return fn / (tp + fn)


def false_positive_rate(fp: float, tn: float) -> float:
    """
    False positive rate (type 1 error).
    When we predict something when it isn’t,
    we are contributing to the false positive rate.
    We want this to be close to 0.

    Calculation:
        false positives / (false positives + true negatives)

    Parameters
    ----------
    fp : float
        number of false positives
    tn : float
        number of true negtives

    Returns
    -------
    float
        flase positive rate

    """
    return fp / (fp + tn)


def true_negative_rate(tn: float, fp: float) -> float:
    """
    True negative rate (specificity).
    When we don't predict something that it is not, the we contribute to
    the true negative rate.
    We want this to be close to 1.

    Parameters
    ----------
    tn : float
        number of true negatives
    fp : float
        the number of false negatives

    Returns
    -------
    float
        true negative rate

    """
    return tn / (tn + fp)


def gather_performance_metrics(
    y_true: list, y_pred: list, model_name: str, best_params: dict
) -> pd.DataFrame:
    """
    Calculates and gathers different metrics to a single pandas dataframe,
    along with the best parameters of the model.
    The metrics are:
        * Accuracy
        * KS (Kolmogorov–Smirnov test)
        * FNR: false negative rate
        * FPR: false positive rate
        * TNR: true negative rate
        * Recall
        * Precision
        * F1

    Parameters
    ----------
    y_true : list
        the true values of y
    y_pred : list
        the predicted y values
    model_name : str
        the name of the model to be used as the index

    Returns
    -------
    pd.DataFrame
        The dataframe with the metrics values

    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # false negative rate
    fnr = false_negative_rate(tp, fn)
    fpr = false_positive_rate(fp, tn)
    tnr = true_negative_rate(tn, fp)

    # true positive rate or sensitivity or recall
    # how many observations out of all positive
    # observations have we classified as positive
    # we want this to be close to 1
    recall = recall_score(y_true, y_pred)

    # positive predictive value
    # how many observations predicted as positive are in fact positive.
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # accuracy, how many observations, both positive and negative,
    # were correctly classified
    accuracy = accuracy_score(y_true, y_pred)
    ks = kstest(y_pred, y_true)[0]
    return pd.DataFrame(
        data=[[accuracy, ks, fnr, fpr, tnr, recall, precision, f1, best_params]],
        columns=[
            "Accuracy",
            "KS",
            "FNR",
            "FPR",
            "TNR",
            "Recall",
            "Precision",
            "F1",
            "Best Parameters"
        ],
        index=[model_name],
    )


def range_inc(
    start: float, stop: float, step: float, inc: float = 1, dec_pl: int = 2
):
    while start < stop:
        yield round(start, dec_pl)
        start += step
        step += start * inc


def get_column(df: pd.DataFrame, col_num: int) -> str:
    """
    Get the column name of the col_num colum (starting from 0).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be used
    col_num : int
        The column number

    Returns
    -------
    str or type of column name
        The column name

    """
    return df.columns[col_num]


# visualization functions
def plot_heatmap(
    df: pd.DataFrame,
    columns: list,
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = "RdBu",
    annot: bool = True,
    figsize: tuple = (30, 30),
    title: str = "Heatmap",
    title_fontsize: int = 18,
    xticks_fontsize: int = 14,
    yticks_fontsize: int = 14,
    annot_fontsize: int = 8,
) -> None:
    """
    Plots a Pearson correlation heatmap for columns containing continuous
    data in a Pandas dataframe and allows for increased appearance control.
    The resulting heatmap is not mirrored. Skips 0 correlations.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe source of the data
    columns : list
        list of the columns to be included
    vmin : float, optional
        minimum correlation value. The default is -1.
    vmax : float, optional
        maximum correlation value. The default is 1.
    cmap : str, optional
        the color scheme to be used. The default is "RdBu".
    annot : bool, optional
        whether or not the heat map will be annotated. The default is True.
    figsize : tuple, optional
        the gaph's dimentions. The default is (30, 30).
    title : str, optional
        title of the graph. The default is 'Heatmap'.
    title_fontsize : int, optional
        font size of title. The default is 18.
    xticks_fontsize : int, optional
        fontsize of x axis ticks. The default is 14.
    yticks_fontsize : int, optional
        fontsize of y axis ticks. The default is 14.
    annot_fontsize : int, optional
        fontsize of annotate values on heatmap boxes. The default is 8.

    Returns
    -------
    None

    """
    plt.figure(figsize=figsize, facecolor="white")
    plt.title(f"{title}", size=title_fontsize)
    corr = df[columns].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(
            corr,
            cmap=cmap,
            mask=mask,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            annot_kws={"fontsize": annot_fontsize},
        )
    plt.xticks(size=xticks_fontsize)
    plt.yticks(size=yticks_fontsize)


def plot_3pca_components(X: np.array, y: np.array) -> plt:
    """
    3D scatterplot of top 3 pca components.

    Parameters
    ----------
    X : np.array
        array-like features data
    y : np.array
        1D array-like target data

    Returns
    -------
    plt
        matplotlib plot

    """
    pca_train = PCA(n_components=3).fit_transform(X)
    
    # 3d scatterplot for components
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([x[0] for x in pca_train], [y[1] for y in pca_train], [z[2] for z in pca_train], c=y)
    return plt


def display_component(
    pca_fitted_model: object,
    num_of_components: int,
    features_list: list,
    component_number: int,
    n_weights_to_display: int = 10,
) -> plt:
    """
    Examine the makeup of each PCA component based on the weightings
    of the original features that are included in the component.
    Note that the components are ordered from smallest to largest
    and so we are getting the correct rows by calling
    num_of_components-component_number to get the component_number
    component.

    Parameters
    ----------
    pca_fitted_model : object
        pca model to be used.
    num_of_components : int
        number of components
    features_list : list
        list of features to be matched with the components.
    component_number : int
        The numbed of component to be displayed
    n_weights_to_display : int, optional
        The number of weights to be displayed. The default is 10.

    Returns
    -------
    matplotlib plot
    """
    # get index of component (last row - component_num)
    row_idx = num_of_components - component_number

    components_makeup = pd.DataFrame(pca_fitted_model.components_)
    # get the list of weights from a row in v, dataframe
    v_1_row = components_makeup.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe
    comps = pd.DataFrame(
        list(zip(v_1, features_list)), columns=["weights", "features"]
    )

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps["abs_weights"] = comps["weights"].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values(
        "abs_weights", ascending=False
    ).head(n_weights_to_display)

    # display using seaborn
    ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(
        data=sorted_weight_data, x="weights", y="features", palette="Blues_d"
    )
    ax.set_title("PCA Component Makeup, Component #" + str(component_number))
    return plt


def create_transformed_df(
    pca_components: list,
    scaled_train_df: pd.DataFrame,
    num_of_components: int,
    n_top_components: int,
) -> pd.DataFrame:
    """
    Return a dataframe of data points with component features.

    Parameters
    ----------
    pca_components : list
        pca somponents to use
    scaled_train_df : pd.DataFrame
        DESCRIPTION.
    num_of_components : int
        The total number of components
    n_top_components : int
        The numbder of top components to use

    Returns
    -------
    pd.DataFrame with the top components
        

    """
    # create new dataframe to add data to
    df = pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in pca_components:
        # get component values for each data point
        components = data.label["projection"].float32_tensor.values
        df = df.append([list(components)])

    # index by county, just like counties_scaled
    df.index = scaled_train_df.index

    # keep only the top n components
    start_idx = num_of_components - n_top_components
    df = df.iloc[:, start_idx:]

    # reverse columns, component order
    return df.iloc[:, ::-1]


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
) -> plt:
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

    Returns
    ------
        matplolib plot
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="g",
        label="Cross-validation score",
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def scatterplot_range_knn_score(
    scores: list, from_: int = 0, to: int = 10, figsize: tuple = (10, 6)
) -> plt:
    """
    Plot metric score of a knn classifier, for a range of K values.

    Parameters
    ----------
    scores : list
        list of float scores, like accuracy or error
    from_ : int, optional
        The k parameter to start from The default is 0.
    to : int, optional
        The last knn parameter. The default is 10.
    figsize : tuple, optional
        The dimentions of the plot. The default is (10, 6).

    Returns
    -------
    plt
        matplotlib plot

    """
    plt.figure(figsize=figsize)
    plt.plot(
        range(from_, to),
        scores,
        color="blue",
        linestyle="dashed",
        marker="o",
        markerfacecolor="red",
        markersize=10,
    )
    plt.title("Accuracy vs. K Value")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    print(
        "Maximum accuracy:-", max(scores), "at K =", scores.index(max(scores))
    )
    return plt
