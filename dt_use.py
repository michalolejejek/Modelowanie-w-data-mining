import numpy as np
from decision_tree import DecisionTree

X = np.random.randint(2, size=(5, 5))
y = np.random.randint(2, size=5)

model = DecisionTree()
model.fit(X, y)

y_pred = model.predict(X)
accuracy = model.accuracy_score(y, y_pred)

print("X Matrix:")
print(X)
print("Etykiety Y:")
print(y)
print("Predykcja:")
print(y_pred)
print("Dokładność modelu:")
print(accuracy)
