import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/resampled_data.csv")
X = df.drop("signal", axis=1)
y = df["signal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# grid search to find optimal value based on accuracy
acc = []
from_ = 1
to = 80
for i in range(from_, to):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train.values)
    pred_i = knn.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, pred_i))

# plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 80),
    acc,
    color="blue",
    linestyle="dashed",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.title("Accuracy vs. K Value")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.text(
    from_ + 1,
    max(acc),
    f"Max accuracy: {round(max(acc), 2)},  K = {acc.index(max(acc))}",
)
plt.savefig("images/knn_benchmark_acc.png")
