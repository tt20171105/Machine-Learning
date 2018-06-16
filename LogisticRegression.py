# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:23:40 2017

@author: tt20171105
"""
import numpy as np
from sklearn import datasets

#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

iris = datasets.load_iris()
x = iris.data[:100]
x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)  #add bias
y = iris.target[:100]

max_iter = 100
eta      = 0.01

w = np.zeros(x.shape[1])
for _ in range(max_iter):
    w_prev = np.copy(w)
    sigma  = sigmoid(np.dot(x, w))
    grad   = np.dot(x.T, (sigma - y))
    w     -= eta * grad
    if np.allclose(w, w_prev):
        break

proba   = sigmoid(np.dot(x, w))
results = (proba > 0.5).astype(np.int)

