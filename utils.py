import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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




