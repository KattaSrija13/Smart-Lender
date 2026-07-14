from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([0, 1, 0, 1])

model = DecisionTreeClassifier()

print("Training...")

model.fit(X, y)

print("Completed")