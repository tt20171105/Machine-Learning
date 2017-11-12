# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:23:40 2017

@author: tt20171105
"""
import numpy as np
import pandas as pd
from sklearn import datasets

INPUT_PATH = "winequality-red.csv"

#データの準備(iris)
def cre_data_iris():
    iris = datasets.load_iris()
    X    = iris.data
    Y    = iris.target
    return X, Y
    
def cre_wine_quality():
    X = pd.read_csv(INPUT_PATH, sep=";")
    Y = np.array(X["quality"])
    X = X.drop("quality", axis=1)
    X = X.apply(lambda x: (x - x.mean()) / x.std())  #標準化
    X = np.array(X)
    return X, Y
    
#####################################
#EMアルゴリズム
def fit(X, alpha, cluster, iter_max=10):
    #次元数
    ndim    = np.size(X, 1)
    #混合比
    weights = np.ones(cluster) / cluster
    #平均
    means   = np.random.uniform(X.min(), X.max(), (ndim, cluster))
    #分散共分散
    covs    = np.repeat(10 * np.eye(ndim), cluster).reshape(ndim, ndim, cluster)
    for i in range(iter_max):
        print("epoch:" + str(i))
        #収束判定用
        params = np.hstack((weights.ravel(), means.ravel(), covs.ravel()))
        #Eステップ
        resps  = expectation(X, ndim, weights, means, covs)
        #Mステップ
        weights, means, covs = maximization(X, alpha, cluster, resps)
        #収束判定
        if np.allclose(params, np.hstack((weights.ravel(), means.ravel(), covs.ravel()))):
            break
    return ndim, weights, means, covs

def gauss(X, ndim, means, covs):
    #逆行列を計算
    precisions = np.linalg.inv(covs.T).T
    #入力データと平均の差を計算
    #     クラスタ１     クラスタ２
    #[[[変数１の差分, 変数１の差分],  #１つ目のデータ
    #  [変数２の差分, 変数２の差分],  #（データの１行目）
    #  [変数３の差分, 変数３の差分]],
    # [[ ・・・ ]]]
    diffs = X[:, :, None] - means
    #   差分 と 分散共分散行列 の内積をとる
    #            [1, 2, 3]          クラスタ１          クラスタ２
    #[A, B ,C] * [4, 5, 6] = [A*1 + A*4 + A*7, ・・・],[ ・・・ ]
    #            [7, 8, 9]
    #ここでは上記を縦にもつ、つまり、
    #    クラスタ１      クラスタ２
    #[[[変数１の内積, 変数１の内積],  #１つ目のデータ
    #  [変数２の内積, 変数２の内積],  #（データの１行目）
    #  [変数３の内積, 変数３の内積]],
    # [[ ・・・ ]]]
    exponents = np.einsum('nik,ijk->njk', diffs, precisions)
    #内積 と 差分をかけて　クラスタ別で各データごとに和を計算する
    #          クラスタ１                 クラスタ２
    #[[変数１～３の内積 × 差分の和, 変数１～３の内積 × 差分の和], #１つ目のデータ
    # [ ・・・ ]]
    exponents = np.sum(exponents * diffs, axis=1)
    #多次元正規分布の式
    #今の平均、分散における、当該データの生成確率
    #       クラスタ１              クラスタ２
    #[[１つ目のデータの生成確率, １つ目のデータの生成確率],
    # [２つ目のデータの生成確率, ２つ目のデータの生成確率],
    # [ ・・・ ]]
    return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(covs.T).T * (2 * np.pi) ** ndim)

def expectation(X, ndim, weights, means, covs):
    #各データの生成確率に混合比をかけて負担率を計算する
    #混合比は各クラスタが全データの何割を負担しているか
    resps  = weights * gauss(X, ndim, means, covs)
    #各データごとに全クラスタの和をとり、それぞれのクラスタの負担率を割る（０～１に正規化）
    #負担率は当該データがどのクラスタから生成されたかの確率
    resps /= resps.sum(axis=-1, keepdims=True)
    return resps

def maximization(X, alpha, cluster, resps):
    #クラスタ別の負担率の和を計算する
    Nk      = np.sum(resps, axis=0)
    #０～１に正規化したものを新たな混合比とする
    weights = Nk / len(X)
    #クラスタ別に各データの値と負担率をかけて和をとる
    #クラスタ別の負担率で割り、その結果を新たな平均値とする
    means   = X.T.dot(resps) / Nk
    diffs   = X[:, :, None] - means
    #差分と負担率をかけて、さらに差分との内積をとる
    #クラスタ別の負担率で割り、その結果を新たな分散共分散とする
    covs    = np.einsum('nik,njk->ijk', diffs, diffs * np.expand_dims(resps, 1)) / Nk
    return weights, means, covs

def predict_proba(X, ndim, weights, means, covs):
    predict = weights * gauss(X, ndim, means, covs)
    return np.sum(predict, axis=-1)

def classify(X, ndim, weights, means, covs):
    joint_prob = weights * gauss(X, ndim, means, covs)
    return np.argmax(joint_prob, axis=1)

##########################################
X, Y = cre_data_iris()     #irisからデータを生成
X, Y = cre_wine_quality()  #赤ワインのデータを取得

ALPHA   = 0.001  #学習率
CLUSTER = 3      #クラスタ数

ndim, weights, means, covs = fit(X, ALPHA, CLUSTER, iter_max=10000)
labels = classify(X, ndim, weights, means, covs)

df = pd.DataFrame(X)
df["Y"]       = Y
df["cluster"] = labels

for cluster in df["cluster"].unique():
    print(df[df["cluster"]==cluster]["Y"].value_counts(sort=False))


"""
A = np.array([[1, 1, 1],
              [2, 2, 2],
              [5, 5, 5]])

B = np.array([[0, 1, 0],
              [1, 1, 0],
              [1, 1, 1]])

np.einsum('ij,jk->ik', A, B)

#array([[ 2,  3,  1],
#       [ 4,  6,  2],
#       [10, 15,  5]])

#A   B   A   B   A   B
#1 * 0 + 1 * 1 + 1 * 1 = 2
#1 * 1 + 1 * 1 + 1 * 1 = 3
#1 * 0 + 1 * 0 + 1 * 1 = 1

#2 * 0 + 2 * 1 + 2 * 1 = 4
#2 * 1 + 2 * 1 + 2 * 1 = 6
#2 * 0 + 2 * 0 + 2 * 1 = 2

np.einsum('ij,jk->ijk', A, B)

#array([[[0, 1, 0],
#        [1, 1, 0],
#        [1, 1, 1]],
#       [[0, 2, 0],
#        [2, 2, 0],
#        [2, 2, 2]],
#       [[0, 5, 0],
#        [5, 5, 0],
#        [5, 5, 5]]])

A = np.array([[[ 1, 2, 3],
               [ 4, 5, 6]],
              [[ 7, 8, 9],
               [10,11,12]]])

B = np.array([[[13,14,15],
               [16,17,18]],
              [[19,20,21],
               [22,23,24]]])

np.einsum('nik,ijk->njk', A, B)

#array([[[ 89, 128, 171],
#        [104, 149, 198]],
#       [[281, 332, 387],
#        [332, 389, 450]]])

#[[[1 * 13 +  4 * 19, 2 * 14 +  5 * 20, 3 * 15 +  6 * 21],
#  [1 * 16 +  4 * 22, 2 * 17 +  5 * 23, 3 * 18 +  6 * 24]],
# [[7 * 13 + 10 * 19, 8 * 14 + 11 * 20, 9 * 15 + 12 * 21],
#  [7 * 16 + 10 * 22, 8 * 17 + 11 * 23, 9 * 18 + 12 * 24]]]

#         [1, 2, 3]
#[A, B .C][4, 5, 6] の内積をとる
#         [7, 8, 9]
"""