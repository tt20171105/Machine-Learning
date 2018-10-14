
# coding: utf-8
import numpy as np
from models    import Sequential
from layers    import Dense, Conv2D, Dropout, Pooling
from optimizer import Adam

from keras.datasets import mnist
from keras.utils    import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train.astype('float32') / 255).reshape(-1, 1, 28, 28)
y_train = np_utils.to_categorical(y_train.astype('int32'), 10)
x_test  = (x_test.astype('float32') / 255).reshape(-1, 1, 28, 28)
y_test  = np_utils.to_categorical(y_test.astype('int32'),  10)

threshold = 1000
x_train = x_train[:threshold]
y_train = y_train[:threshold]
x_test  = x_test[:threshold]
y_test  = y_test[:threshold]

seed  = 15
model = Sequential(seed=seed)
model.add(Conv2D(32, (5,5), activation="relu", inputs_shape=x_train.shape[1:]))
model.add(Pooling((2,2)))
model.add(Conv2D(16, (3,3), activation="relu"))
model.add(Pooling((2,2)))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(loss      = "categorical_crossentropy",
              optimizer = Adam(),
              metric    = "accuracy")

model.fit(x_train=x_train, t_train=y_train, x_test=x_test, t_test=y_test,
          batch_size=128, epochs=10, output_num=1)

#誤差をプロット
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(model.history_train[0])
plt.plot(model.history_test[0])
plt.title("loss")
plt.legend(["train","test"])
plt.show()
plt.plot(model.history_train[1])
plt.plot(model.history_test[1])
plt.title(model.metric_name)
plt.legend(["train","test"])

