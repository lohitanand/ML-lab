import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))

plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Classification")
plt.show()