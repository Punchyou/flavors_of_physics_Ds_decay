import pandas as pd
import numpy as np
import seaborn as sns

# TODO add heatmap: seaborn.heatmap(data, *, vmin=None, vmax=None, cmap=None,
# center=None, robust=False, annot=None, fmt='.2g', annot_kws=None,
# linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None,
# square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None,
# **kwargs)

# exploration
a = pd.read_csv('data/training.csv')
a.set_index('id', inplace=True)
a['signal'].values_counts()
a.corr()
a.columns[(a.corr()['LifeTime']>0.5).values]

