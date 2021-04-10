import pandas as pd
from utils import subplot_correlation_matrix, plot_heatmap, display_component, create_transformed_df
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


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


# now create the reansformed training set
display_component(
    pca_fitted_model=pca_no_comps_train_model,
    num_of_components=19,
    features_list=X_train.columns,
    component_number=2,
    n_weights_to_display=15)

# TODO check how to create dimensionality-reduced data - what format should the
# pca components be
create_transformed_df

# TODO try dimnetionality_reduction_knn.py function for pca dn knn


# find optimal value based on error

from_, to = 40, 80
error_rate = pd.DataFrame(index=range(from_, to), columns=['rate'])
for i in range(from_, to):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.loc[] = error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(from_, to), error_rate.values, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate), "at K =",
      error_rate[error_rate['rate']==min(error_rate)].index)
plt.show()

# find optimal k values based on accuracy
acc = []
# Will take some time
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10,6))
plt.plot(range(40,100),acc,color = 'blue',linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))




"""
How to choose the best n components for pca:
* PCA project the data to less dimensions, so we need to scale the data
beforehand.

* A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. This can be determined by looking at the cumulative explained variance ratio as a function of the number of components
This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components.

* From the pca graph, looks like that the data are described from 3 variables.

Sources:
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
Data Science from Scratch
Machine learning with python
Feature Egnineering

"""
