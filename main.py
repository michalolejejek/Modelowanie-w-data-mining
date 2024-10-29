import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from decision_tree import DecisionTree

X = np.random.randint(2, size=(5, 5))
y = np.random.randint(2, size=5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)

model = DecisionTree()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = model.accuracy_score(y_test, y_pred)

print("X Matrix:")
print(X)
print("Etykiety Y:")
print(y_test)
print("Predykcja:")
print(y_pred)
print("Dokładność modelu:")
print(accuracy)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()
