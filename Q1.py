import pandas as pd

data = pd.DataFrame({
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Enjoy': ['Yes', 'Yes', 'No', 'Yes']
})

h = ['0'] * (len(data.columns) - 1)

for i in range(len(data)):
    if data.iloc[i, -1] == 'Yes':
        for j in range(len(h)):
            if h[j] == '0':
                h[j] = data.iloc[i, j]
            elif h[j] != data.iloc[i, j]:
                h[j] = '?'

print("Final Hypothesis:", h)