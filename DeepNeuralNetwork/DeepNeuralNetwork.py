# coding: utf-8
import copy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets

from models import Sequential
from layers import Dense, Dropout
from optimizer import Adam

def one_hot_encoding(t):
    t = np.array(t).reshape(1, -1)
    t = t.transpose()
    encoder = OneHotEncoder(n_values=max(t)+1)
    t = encoder.fit_transform(t).toarray()
    return t

iris   = datasets.load_iris()
iris_x = iris.data
iris_t = iris.target
def create_data_category():
    t = copy.deepcopy(iris_t)
    t = one_hot_encoding(t)
    return iris_x, iris_x.shape[1], t, t.shape[1]

def create_data_numeric(target):
    c = np.ones(iris_x.shape[1], dtype=bool)
    c[target] = False
    x = iris_x[:, c]
    t = iris_x[:, target].reshape(-1,1)
    return x, x.shape[1], t, 1

#irisからデータを生成
#x, inputs_shape, t, outputs_shape = create_data_category()
#loss   = "categorical_crossentropy"
#metric = "accuracy"
#last_layer_activation = "softmax"
x, inputs_shape, t, outputs_shape = create_data_numeric(3)
loss   = "mean_squared_error"
metric = "rmse"
last_layer_activation = "identify"

seed  = 15
model = Sequential(seed=seed)
model.add(Dense(10, activation="relu", inputs_shape=inputs_shape))
model.add(Dense(10, activation="relu"))
model.add(Dense(outputs_shape, activation=last_layer_activation))
model.compile(loss     =loss,
              optimizer=Adam(),
              metric   =metric)

train_x, test_x, train_t, test_t = train_test_split(x, t, 
                                                    test_size=0.3,
                                                    random_state=seed)
model.fit(train_x, train_t, test_x, test_t,
          epochs=1000, batch_size=50)

#誤差をプロット
import matplotlib.pyplot as plt

plt.plot(model.history_train[0])
plt.plot(model.history_test[0])
plt.title("loss")
plt.legend(["train","test"])

plt.plot(model.history_train[1])
plt.plot(model.history_test[1])
plt.title(model.metric_name)
plt.legend(["train","test"])
