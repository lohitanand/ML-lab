import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price': [30, 45, 50, 65, 80]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("Predicted Price:", pred)

plt.scatter(y, pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.show()