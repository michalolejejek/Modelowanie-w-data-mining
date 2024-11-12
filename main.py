import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from decision_tree import DecisionTree

X = np.random.randint(2, size=(5, 5))
y = np.random.randint(2, size=5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)

#Nasz model
model_custom = DecisionTree()
model_custom.fit(X_train, y_train)
y_pred_custom = model_custom.predict(X_test)
accuracy_custom = model_custom.accuracy_score(y_test, y_pred_custom)

#Model z Scikit learn
model_sklearn = DecisionTreeClassifier(random_state=11)
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)


print("X Matrix:")
print(X)
print("Etykiety Y:")
print(y_test)
print("\n--- Wyniki naszej implementacji ---")
print("Predykcja:", y_pred_custom)
print("Dokładność modelu:", accuracy_custom)

print("\n--- Wyniki modelu scikit-learn ---")
print("Predykcja:", y_pred_sklearn)
print("Dokładność modelu:", accuracy_sklearn)

#Macierz konfuzji dla obu modeli
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#Macierz konfuzji dla własnej implementacji
cm_custom = confusion_matrix(y_test, y_pred_custom)
sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
axes[0].set_title("Confusion Matrix - Our Model")
axes[0].set_xlabel("Predicted labels")
axes[0].set_ylabel("True labels")

#Macierz konfuzji dla modelu scikit-learn
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1])
axes[1].set_title("Confusion Matrix - scikit-learn Model")
axes[1].set_xlabel("Predicted labels")
axes[1].set_ylabel("True labels")

plt.tight_layout()
plt.show()
