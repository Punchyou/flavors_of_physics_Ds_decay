import matplotlib.pyplot as plt
import mypy
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler

from utils import correlation_heatmap, plot_3pca_components


def main():
    df = pd.read_csv("data/resampled_data.csv")
    y = df["signal"]
    X = df.drop("signal", axis=1)

    # nan values
    len(df) - len(df.dropna())

    # check data types of columns
    print(df.dtypes)

    # check if the dataset is balanced
    # check couts of values
    df["signal"].value_counts()  # 0: 8205, 1: 8205
    sns.countplot(x="signal", data=df)
    plt.savefig("images/signal_value_counts.png")

    # check distributions
    X.hist(figsize=(60,30))
    plt.tight_layout()
    plt.savefig("images/features_distributions.png")

    # check correlations
    # describe
    print(df.describe())

    # plot correlation heatmap
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(data=df_scaled, columns=df.columns)
    correlation_heatmap(
        df=df_scaled, columns=df.columns, figsize=(20, 20), annot_fontsize=6, title=""
    )
    plt.tight_layout()
    plt.savefig("images/features_correlation_heatmap.png")

    # Dimentionality Reduction
    # scaling is required for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # apply pca transform
    plot_3pca_components(X=X_scaled, y=y)
    # I have already saved this plot. As this is a 3d plot, it needs to be
    # turned before saved, to get a better angle that shows the classes more
    # clearly than the default angle
    # plt.savefig("plot/pca_binary_scatter_3d_plot.png")


if __name__ == "__main__":
    main()
