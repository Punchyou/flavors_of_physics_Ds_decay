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


# TODO check doscstring + add params
def display_component(pca_fitted_model, num_of_components, features_list,
                      component_number, n_weights_to_display=10):
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
    return df.iloc[:, ::-1]i




