import pandas as pd
import math
from collections import Counter

# Dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes']
})

# Entropy function
def entropy(y):
    counts = Counter(y)
    total = len(y)
    e = 0

    for count in counts.values():
        p = count / total
        e -= p * math.log2(p)

    return e

# Calculate entropy
total_entropy = entropy(data['Play'])

# Calculate information gain
feature = 'Outlook'
gain = total_entropy

for value in data[feature].unique():
    subset = data[data[feature] == value]
    gain -= (len(subset) / len(data)) * entropy(subset['Play'])

print("Entropy:", total_entropy)
print("Information Gain:", gain)