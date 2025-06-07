from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def multilayer_perceptron_classifier():
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Y Pred:', y_pred)
    print('Y test:', y_test)
    return

# multilayer_perceptron_classifier()

def multilayer_perceptron_regressor():
    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=1500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    meansq = mean_squared_error(y_pred,y_test)
    print('MSE:', meansq)
    print('Y Pred:', y_pred)
    print('Y test:', y_test)
    return

multilayer_perceptron_regressor()