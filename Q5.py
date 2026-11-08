import pandas as pd
from collections import Counter

# Dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes']
})

# Gini function
def gini(y):
    counts = Counter(y)
    total = len(y)
    g = 1

    for count in counts.values():
        p = count / total
        g -= p ** 2

    return g

# Calculate Gini Index
feature = 'Outlook'
gini_index = 0

for value in data[feature].unique():
    subset = data[data[feature] == value]

    gini_index += (len(subset) / len(data)) * gini(subset['Play'])

print("Gini Index:", gini_index)