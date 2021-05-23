import pandas as pd
from utils import plot_heatmap, display_component, plot_3pca_components
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import mypy

def main():
    df = pd.read_csv("data/resampled_data.csv")
    y = df["signal"]
    X = df.drop("signal", axis=1)

    # nan values
    len(df) - len(df.dropna())

    # check data types of columns
    df.dtypes

    # check distributions
    X.hist(figsize=(60, 35))
    plt.savefig("plots/features_distributions.png")
    
    # check if the dataset is balanced
    # check couts of values
    df['signal'].value_counts() #0: 8205, 1: 8205
    sns.countplot(x = 'signal', data=df)
    plt.savefig("plots/signal_value_counts.png")
    
    # check correlations
    # describe
    df.describe()

    # plot correlation heatmap
    plot_heatmap(df=df, columns=df.columns, figsize=(20, 16), annot_fontsize=6)
    plt.savefig("plots/features_correlation_heatmap.png")

    # Dimentionality Reduction
    # scaling is required for PCA
    rob_scaler = RobustScaler()
    X_rob_scaled = rob_scaler.fit_transform(X)

    # apply pca transform
    plot_3pca_components(X=X_rob_scaled, y=y)
    plt.savefig("plot/pca_binary_scatter_3d_plot.png")

