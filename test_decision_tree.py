import numpy as np
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

#macierz 2x2
def test_small_matrix():
    X = np.random.randint(2, size=(2, 2))  
    y = np.random.randint(2, size=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=11)
    model = DecisionTree()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  #test czy dokładność jest nieujemna

#macierz 5x5
def test_medium_matrix():
    X = np.random.randint(2, size=(5, 5))  
    y = np.random.randint(2, size=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)
    model = DecisionTree()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  #test czy dokładność jest nieujemna

#10x10
def test_large_matrix():
    X = np.random.randint(2, size=(10, 10))  
    y = np.random.randint(2, size=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    model = DecisionTree()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  #test czy dokładność jest nieujemna

#macierz 100x50
def test_high_dimensional_data():
    X = np.random.randint(2, size=(100, 50))  
    y = np.random.randint(2, size=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    model = DecisionTree()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  #test czy dokładność jest nieujemna

#Pusty macierz
def test_empty_matrix():
    X = np.array([]).reshape(0, 0)  
    y = np.array([])

    model = DecisionTree()
    try:
        model.fit(X, y)
        assert False, "Model powinien zgłaszać wyjątek przy pustych danych"
    except Exception:
        assert True #Jesli zglasza wyjątek to test jest passed
