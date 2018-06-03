# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:18:12 2017

@author: tt20171105
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

class Regression():
    
    def __init__(self, 
                 reg_type="multiple_regression"):
        
        if   reg_type=="mulitiple_regression":
            self.base_function = self._multiple_regression
        elif reg_type=="polynominal_regression":
            self.base_function = self._polynominal_regression
        else:
            self.base_function = self._multiple_regression
            
        self.f = None
        self.w = None
        self.rmse_max_idx = None
            
    #最小二乗法
    def _least_square_method(self, x, y):
        #バイアス項を追加
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        #w=(x^T*x)^(−1)*x^T*yを計算
        #xの内積をとり逆行列を求める
        xx_inv = np.linalg.inv(np.dot(x.T, x))  #m*n, n*m = m*m
        #逆行列とxの内積を求める
        xx_inv_x = np.dot(xx_inv, x.T)  #m*m, m*n = m*n
        #逆行列の内積とyの内積を求める
        w = np.dot(xx_inv_x, y)  #m*n, n*1 = m*1
        def f(x):
            return (x * w[-x.shape[1]:]).sum(1) + w[0]    
        return f, w

    #重回帰分析
    def _multiple_regression(self, x, y):
        f, w = self._least_square_method(x, y)
        return f, w

    #多項式回帰分析
    def _polynominal_regression(self, x, y):
        results = []
        rmses   = []
        for i in range(x.shape[1]):
            x_poly = self.polynominal(x, i)
            f, w   = self._least_square_method(x_poly, y)
            results.append([f, w])
            rmses.append(rmse(f(x_poly), y))
        #RMSEが最大になる結果のインデックス
        self.rmse_max_idx = np.argmax(rmses)
        result = results[self.rmse_max_idx]
        return result[0], result[1]
    
    def polynominal(self, x, x_idx):
        x_polynominal = np.zeros((x.shape[0], self.degree))
        for i in range(self.degree):
            x_polynominal[:, i] = x[:, x_idx] ** (i+1)
        return x_polynominal

    def rmse(self, y, t):
        rmse = np.sqrt(np.mean((y - t) ** 2))
        return rmse
    
    def fit(self, x, t, degree=2):
        self.degree = degree
        f, w = self.base_function(x, t)
        self.f = f
        self.w = w

y_idx = 2

iris = datasets.load_iris()
x = np.delete(iris.data, y_idx, 1)
y = iris.data[:, y_idx]

#多項式回帰分析
rg = Regression("polynominal_regression")
rg.fit(x, y, 3)

y_hat = rg.f(rg.polynominal(x, rg.rmse_max_idx))
print(rg.rmse(y_hat, y))

plt.plot(y_hat)
plt.plot(y)

#重回帰分析
rg = Regression()
rg.fit(x, y)

y_hat = rg.f(x)
print(rg.rmse(y_hat, y))

plt.plot(y_hat)
plt.plot(y)

