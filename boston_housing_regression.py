import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
df = pd.read_csv("HousingData.csv")
df = df.fillna(0)
X, y = df.values[1:, :-1], df.values[1:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_features = X_train.shape[1]
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
error = model.evaluate(X_test, y_test, verbose=1)
print('MSE %.3f RMSE %.3f' % (error, np.sqrt(error)))
row = np.array([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]])
yhat = model.predict(row)
print('Predicted %.3f' % yhat)
