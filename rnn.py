from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import numpy as np
from numpy import asarray
from numpy import sqrt
train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
windowSize, X_train, y_train = 3, [], []
for index in range(len(train)-windowSize):
    X_train.append(train[index:index+windowSize])
    y_train.append(train[index+windowSize])
print(X_train)
print(y_train)
model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=500)
mse, mae = model.evaluate(X_train, y_train, verbose=1)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
row = np.array([[2, 3, 4]])
row1 = row.reshape(1, windowSize, 1)
print(row1)
yhat = model.predict([[row1]])
print(yhat)
row2 = asarray([2, 3, 4]).reshape((1, windowSize, 1))
yhat = model.predict(row2)
print(yhat)