import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
windowSize, X_train, y_train = 3, [], []
for index in range(len(train)- windowSize):
    X_train.append(train[index:index+windowSize])
    y_train.append(train[index+windowSize])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(len(X_train), 3, 1)
model = Sequential()
model.add(SimpleRNN(20, input_shape=(3, 1), return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=500)
mse, mae = model.evaluate(X_train, y_train, verbose=1)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, np.sqrt(mse), mae))
row = np.array([[2, 3, 4]])
row1 = row.reshape(1, windowSize, 1)
yhat = model.predict([row1])
row2 = np.asarray([2, 3, 4]).reshape((1, windowSize, 1))
yhat = model.predict(row2)
print(yhat)