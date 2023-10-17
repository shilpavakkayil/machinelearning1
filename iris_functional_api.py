import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
df = pd.read_csv("iris_dataset.csv")
X, y = df.values[1:, :-1], df.values[1:, -1]
X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_features = X_train.shape[1]
x_in = Input(shape=(n_features,))
x2 = Dense(10, activation='relu', kernel_initializer='he_normal')(x_in)
x3 = Dense(10, activation='relu', kernel_initializer='he_normal')(x2)
x_out = Dense(3, activation='softmax')(x3)
model = keras.Model(inputs=x_in, outputs=x_out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)
# make a prediction
row1 = [5.1,3.5,1.4,0.2]
row2 = np.array([[5.1,3.5,1.4,0.2]])
yhat = model.predict([row2])
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))


