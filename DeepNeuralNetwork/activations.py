
# coding: utf-8
import numpy as np

#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#双曲線正接関数
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

#ランプ関数
def relu(x):
    return x * (x > 0)
def d_relu(x):
    return 1 * (x > 0)

#ソフトプラス関数
def softplus(x):
    return np.log(1 + np.exp(x))
def d_softplus(x):
    return 1 / (1 + np.exp(-x))

#恒等関数
def identify(x):
    return x
def d_identify(x):
    return 1

#ソフトマックス関数
def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def d_softmax(x):
    return softmax(x) * (1 - softmax(x))

def get_activation_function(activation_function):
    activation   = {"relu"     : relu,
                    "tanh"     : tanh,
                    "sigmoid"  : sigmoid,
                    "softplus" : softplus,
                    "identify" : identify,
                    "softmax"  : softmax}
    return activation[activation_function]

def get_d_activation_function(d_activation_function):
    d_activation = {"relu"     : d_relu,
                    "tanh"     : d_tanh,
                    "sigmoid"  : d_sigmoid,
                    "softplus" : d_softplus,
                    "identify" : d_identify,
                    "softmax"  : None}
    return d_activation[d_activation_function]

