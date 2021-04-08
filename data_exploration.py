import pandas as pd
from utils import subplot_correlation_matrix, plot_heatmap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# exploration
# load data
train_df = pd.read_csv('data/training.csv', index_col='id')
y = train_df['signal']
X = train_df.drop('signal', axis=1)

# test_df = pd.read_csv('data/test.csv', index_col='id')
# 
# # kepp only common columns in both train and test sets
# common_cols = np.intersect1d(train_df.columns, test_df.columns)
# train_df = train_df[common_cols]
# test_df = test_df[common_cols]

# check couts of values
train_df['signal'].value_counts()

# check correlations
train_df.corr()
train_df.columns[(train_df.corr()['LifeTime'] > 0.5).values]

# nan values
len(train_df) - len(train_df.dropna())

# check data types of columns
train_df.dtypes

# describe
train_df.describe()

subplot_correlation_matrix(train_df, (30, 30))
plt.show()

plot_heatmap(df=train_df, columns=train_df.columns, figsize=(10, 8), annot_fontsize=6)
plt.show()

# split in X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# scaling is required for pca
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Dimentionality Reduction
# calculate parameters
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
# looks like that the data are described by 3 variables

# apply pca transform
pca_train = PCA(n_components=3).fit_transform(X_train_scaled)
pca_test = PCA(n_components=3).fit_transform(X_test_scaled)

# 3d scatterplot for components
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([x[0] for x in pca_train], [y[1] for y in pca_train], [z[2] for z in
                                                            pca_train])
plt.show()

# TODO based on the plot above, go for DBSCAN
# Second way of choosing the number of components
# define the percentage of the total variance that the data cover in the
# dataset
pca_no_comps_train_model = PCA(n_components=0.95) # 95 percent of variance
pca_no_comps_train_comps = pca_no_comps_train_model.fit_transform(X_train_scaled)

# check the proportion of the data's variance that lies along the axis for each
# principal component
components_95per_var = pca_no_comps_train_model.explained_variance_ratio_
# check how many they are
len(components_95per_var) # 19 coponents

# check how many complenents describe how much variance for 9 components
components_95per_var[:9].sum() # 0.82
# TODO do the above for test set too

# TODO check notebook on kmeans after pca to see the transformation


def display_component(pca_fitted_model, num_of_components, features_list,
                      component_number, n_weights_to_display=10):

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

# now create the reansformed training set
# create dimensionality-reduced data
def create_transformed_df(pca_components, scaled_train_df, num_of_components, n_top_components):
    ''' Return a dataframe of data points with component features. 
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.        
     '''
    # create new dataframe to add data to
    df=pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in pca_components:
        # get component values for each data point
        components=data.label['projection'].float32_tensor.values
        counties_transformed=counties_transformed.append([list(components)])

    # index by county, just like counties_scaled
    df.index=scaled_train_df.index

    # keep only the top n components
    start_idx = num_of_components - n_top_components
    df = df.iloc[:,start_idx:]
    
    # reverse columns, component order     
    return counties_transformed.iloc[:, ::-1]
    










"""
How to choose the best n components for pca:
* PCA project the data to less dimensions, so we need to scale the data
beforehand.

* A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. This can be determined by looking at the cumulative explained variance ratio as a function of the number of components
This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components.

* From the pca graph, looks like that the data are described from 3 variables.

Sources:
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

"""
