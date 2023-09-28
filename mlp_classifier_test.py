from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
model = MLPClassifier(random_state=1, max_iter=300)
model.fit(X_train, y_train)
print(X_test[:5, :])
print(model.predict_proba(X_test[:1]))
print(model.predict(X_test[:5, :]))
print(model.score(X_test, y_test))
