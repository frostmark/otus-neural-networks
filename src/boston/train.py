from __future__ import print_function

import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 50


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# preparing data
x_train_mean = x_train.mean(axis=0)
print(f'x_train mean: {x_train_mean}')
x_train = x_train - x_train_mean

x_train_std = x_train.std(axis=0)
print(f'x_train_std: {x_train_std}')
x_train = x_train / x_train_std

x_test = x_test - x_train_mean
x_test = x_test / x_train_std
print(x_train[0])


model = Sequential()
model.add(Dense(13, activation='relu', input_dim=13))
model.add(Dense(13, activation='relu', input_dim=13))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(optimizer='rmsprop',
            loss='mse',
            metrics=['mae'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Mean absolute error:', score[1])
