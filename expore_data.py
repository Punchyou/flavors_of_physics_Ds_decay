import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef, plot_precision_recall_curve
from sklearn.pipeline import make_pipeline
import sklearn as sk
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import mypy

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
sns.countplot(x = 'signal', data=train_df)
plt.show()

# check correlations
train_df.corr()
train_df.columns[(train_df.corr()['LifeTime'] > 0.5).values]

# nan values
len(train_df) - len(train_df.dropna())

# check data types of columns
train_df.dtypes

# describe
train_df.describe()

utils.subplot_correlation_matrix(train_df, (30, 30))
plt.show()

utils.plot_heatmap(df=train_df, columns=train_df.columns, figsize=(10, 8), annot_fontsize=6)
plt.show()

# split in X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
utils.display_component(
    pca_fitted_model=pca_no_comps_train_model,
    num_of_components=19,
    features_list=X_train.columns,
    component_number=2,
    n_weights_to_display=15)

