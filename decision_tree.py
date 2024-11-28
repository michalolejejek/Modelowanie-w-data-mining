import numpy as np

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        # Buduje drzewo decyzyjne na podstawie danych treningowych X i y
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        # Sprawdza, czy wszystkie etykiety są takie same
        if len(set(y)) == 1:
            return y[0]
        # Sprawdza czy nie ma już więcej cech do podziału
        if X.size == 0:
            return np.bincount(y).argmax()

        # Znajduje najlepszą cechę do podziału
        best_feature = self._best_split(X, y)
        tree = {best_feature: {}}
        for value in [0, 1]:
            # Tworzy podzbiory danych na podstawie wartości cechy
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            # Rekurencyjnie buduje drzewo dla podzbiorów
            tree[best_feature][value] = self._build_tree(sub_X, sub_y)
        return tree

    def _best_split(self, X, y):
        # Znajduje cechę z najniższym wskaźnikiem Gini
        best_feature = 0
        best_gini = float('inf')
        for feature in range(X.shape[1]):
            gini = self._gini_index(X[:, feature], y)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
        return best_feature

    def _gini_index(self, feature, y):
        m = len(y)
        gini = 0.0
        for value in [0, 1]:
            sub_y = y[feature == value]
            if len(sub_y) == 0:
                continue
            score = np.sum(sub_y == 1) / len(sub_y)
            gini += (1.0 - score ** 2 - (1 - score) ** 2) * (len(sub_y) / m)
        return gini

    def predict(self, X):
        # Przewiduje etykiety dla danych wejściowych X
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        # Przewiduje etykietę dla pojedynczego przykładu
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        return self._predict_single(x, tree[feature][x[feature]])

    def accuracy_score(self, y_true, y_pred):
        # Dokładność modelu
        return np.sum(y_true == y_pred) / len(y_true)

# Przykładowe użycie
if __name__ == "__main__":
    # Dane treningowe
    X = np.array([
        [0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 0, 0]
    ])
    y = np.array([0, 1, 1, 0, 1])

    # Tworzenie i trenowanie modelu
    model = DecisionTree()
    model.fit(X, y)

    # Przewidywanie
    X_test = np.array([
        [0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0]
    ])
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")

    # Ocena modelu
    y_test = np.array([0, 1])
    accuracy = model.accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
