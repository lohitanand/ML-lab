import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict output
pred = model.predict(X)

print("Predicted Values:", pred)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Plot graph
plt.scatter(X, y, color='blue')
plt.plot(X, pred, color='red')

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Simple Linear Regression")
plt.show()