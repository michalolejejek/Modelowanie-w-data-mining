import numpy as np
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from decision_tree import DecisionTree

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
y_iris = (y_iris == 0).astype(int)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_iris = scaler.fit_transform(X_train_iris)
X_test_iris = scaler.transform(X_test_iris)
dt_iris = DecisionTree()
dt_iris.fit(X_train_iris, y_train_iris)
y_pred_iris = dt_iris.predict(X_test_iris)
accuracy_iris = dt_iris.accuracy_score(y_test_iris, y_pred_iris)
print(f'Iris Dataset Accuracy: {accuracy_iris:.2f}')

mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist.data, mnist.target.astype(int)
y_mnist = (y_mnist == 0).astype(int)
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)
X_train_mnist = scaler.fit_transform(X_train_mnist)
X_test_mnist = scaler.transform(X_test_mnist)
dt_mnist = DecisionTree()
dt_mnist.fit(X_train_mnist, y_train_mnist)
y_pred_mnist = dt_mnist.predict(X_test_mnist)
accuracy_mnist = dt_mnist.accuracy_score(y_test_mnist, y_pred_mnist)
print(f'MNIST Dataset Accuracy: {accuracy_mnist:.2f}')

#wynik dla datasetu Iris
print(f'Dokladnosc dla datasetu Iris: {accuracy_iris:.2f}')
#wynik dla datasetu MNIST
print(f'Dokladnosc dla datasetu MNIST: {accuracy_mnist:.2f}')
