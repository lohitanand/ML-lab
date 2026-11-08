import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

print("Cluster Labels:", model.labels_)
print("Cluster Centers:", model.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            marker='X', s=200, color='red')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()