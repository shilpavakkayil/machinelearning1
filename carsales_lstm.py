import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        #print(seq_x, seq_y, end_ix)
        X.append(seq_x)
        y.append(seq_y)
    return np.asarray(X), np.asarray(y)


df = pd.read_csv('monthly-car-sales.csv', header=0, index_col=0)
values = df.values.astype('float32')
n_steps = 5
X, y = split_sequence(values, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))
n_test = 12
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
model = Sequential()
model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps, 1)))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=350, batch_size=32, verbose=2, validation_data=(X_test, y_test))
mse, mae = model.evaluate(X_test, y_test, verbose=1)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, np.sqrt(mse), mae))
row = np.asarray([18024.0, 16722.0, 14385.0, 21342.0, 17180.0]).reshape((1, n_steps,1))
yhat = model.predict(row)
print('Predicted: %.3f' % (yhat))
