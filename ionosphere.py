from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import numpy as np
if __name__ == '__main__':
    df = read_csv("ionosphere.csv", header=None)
    print(df.shape)
    X, y = df.values[:, :-1], df.values[:, -1]
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    n_features = X_train.shape[1]
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    # predict test set
    predict_x = model.predict(X_test)
    yhat = np.argmax(predict_x, axis=1)
    #yhat = model.predict_classes(X_test)
    # evaluate predictions
    score = accuracy_score(y_test, yhat)
    print('Accuracy: %.3f' % score)
    # plot learning curves
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.show()
