# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:23:40 2017

@author: tt20171105
"""
import numpy as np
from sklearn import datasets

ALPHA = 0.01   #学習率
EPOCH = 10000  #エポック数

#データの準備(iris)
def cre_data_iris():
    iris = datasets.load_iris()
    X    = iris.data[0:100]
    Y    = iris.target[0:100]
    Y    = np.array([[y] for y in Y])
    dimension = len(X[0])
    return X, Y, dimension

#シグモイド関数
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

#irisからデータを生成
X, Y, dimension = cre_data_iris()
#重みの初期化
W = np.random.random_sample((1, dimension))

#ロジスティック回帰
#最急降下法で重みを求める
for i in range(EPOCH):
    theta = sum(X * (sigmoid(np.inner(X, W)) - Y))
    W     = W - ALPHA * theta

for i in range(len(X)):
    p = sigmoid(np.inner(X[i], W))[0]
    if p < 0.5:
        print("predict:0 result:" + str(Y[i][0]) + " probability:" + str(round(p,5)))
    else:
        print("predict:1 result:" + str(Y[i][0]) + " probability:" + str(round(p,5)))
