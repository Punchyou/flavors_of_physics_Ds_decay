import pandas as pd
from utils import plot_heatmap, plot_3pca_components
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
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
    X.hist(figsize=(60, 50))
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
    plot_heatmap(df=df, columns=df.columns, figsize=(20, 20), annot_fontsize=6)
    plt.savefig("plots/features_correlation_heatmap.png")

    # Dimentionality Reduction
    # scaling is required for PCA
    rob_scaler = RobustScaler()
    X_rob_scaled = rob_scaler.fit_transform(X)

    # apply pca transform
    plot_3pca_components(X=X_rob_scaled, y=y)
    # I have already saved this plot. As this is a 3d plot, it needs to be
    # turned before saved, to get a better angle that shows the classes more
    # clearly than the default angle
    #plt.savefig("plot/pca_binary_scatter_3d_plot.png")

if __name__ == "__main__":
    main()