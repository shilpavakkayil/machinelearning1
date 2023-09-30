from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
reg = MLPRegressor(random_state=1, max_iter=500)
reg.fit(X_train,y_train)
print(reg.loss)
print(reg.activation)
print(reg.predict(X_test[:2]))
print(reg.score(X_test, y_test))