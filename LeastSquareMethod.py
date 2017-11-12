# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:18:12 2017

@author: tt20171105
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

#最小二乗法
def l_s_method(_df, target_col):
    t   = _df[target_col]
    phi = _df.drop(target_col, axis=1)
    #w=(Φ^T*Φ)^(−1)*Φ^T*tを計算
    ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi)), phi.T), t)
    def f(ser):
        y = 0
        for i, w in enumerate(ws): y += w * ser[i]
        return y
    return f, ws

#平方根平均二乗誤差の計算
def rmse(_df, _cols, _target_var, f):
    err = 0.0
    for i in range(len(_df)):
        x = _df.loc[i, _cols]
        y = _df.loc[i, _target_var]
        err += 0.5 * (y - f(x)) ** 2
    return np.sqrt(2 * err / len(_df))

def main():
    iris = datasets.load_iris()
    cols = ["s_len","s_wid","p_len","p_wid"]
    df   = pd.DataFrame(iris.data, columns=cols)

    target_var  = cols[3]
    cols.remove(target_var)
    f, ws = l_s_method(df, target_var)
    
    list_result = []
    for i in range(len(df)):
        result = f(df.loc[i, cols])
        list_result.append([result, df.loc[i, target_var]])
        print(result, df.loc[i, target_var])

    #RMSEの算出
    rmse(df, cols, target_var, f)
    #予測結果の表示
    plt.plot(list_result)

if __name__ == '__main__':
    main()
