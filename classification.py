from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import numpy as np
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=1)
n_features = X.shape[1]
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1,activation='sigmoid'))
sgd = SGD(learning_rate=0.001, momentum=0.8)
model.compile(optimizer=sgd, loss='binary_crossentropy')
model.fit(X, y, epochs=100, batch_size=32, verbose=1, validation_split=0.3)
model.save('model.h5')
model = load_model('model.h5')
row = np.array([[1.91518414, 1.14995454, -1.52847073, 0.79430654]])
yhat = model.predict([row])
print('predicted %.3f' % yhat[0])

