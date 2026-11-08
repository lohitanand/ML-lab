import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

model = DecisionTreeClassifier()

params = {
    'max_depth': [2, 3, 4, 5],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(model, params, cv=5)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

plot_tree(grid.best_estimator_, filled=True)
plt.show()