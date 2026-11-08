import numpy as np
from sklearn.neural_network import MLPClassifier

# Dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation='tanh',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

# Train model
model.fit(X, y)

# Predict output
pred = model.predict(X)

print("Predictions:", pred)
print("Actual Output:", y)
print("Accuracy:", model.score(X, y))