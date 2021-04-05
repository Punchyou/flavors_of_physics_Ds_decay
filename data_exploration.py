import pandas as pd
from utils import subplot_correlation_matrix, plot_heatmap

# TODO add heatmap: seaborn.heatmap(data, *, vmin=None, vmax=None, cmap=None,
# center=None, robust=False, annot=None, fmt='.2g', annot_kws=None,
# linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None,
# square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None,
# **kwargs)

# exploration
train_df = pd.read_csv('data/training.csv')
train_df.set_index('id', inplace=True)

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

plot_heatmap(df=train_df, columns=train_df.columns, figsize=(10, 8), annot_fontsize=6)

test_df = pd.read_csv('data/test.csv')

