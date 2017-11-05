# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:02:06 2017

@author: Tamura
"""
import numpy as np
import pandas as pd
from sklearn import datasets

#ハイパーパラメータ
UNIT_NUM      = 100    #隠れ層のユニット数
EPOCH         = 1000   #エポック数
LEARNING_RATE = 0.001  #学習率

#データの準備(iris)
def cre_data_iris():
    #目的変数をベクトル化
    def Y_vectorize(_X, _curt_Y):
        list_Y   = sorted(_X.Y.unique())
        vector_Y = []
        for curt_Y in list_Y:
            if curt_Y == _curt_Y:
                vector_Y.append(1)
            else:
                vector_Y.append(0)
        return vector_Y

    iris = datasets.load_iris()
    data = iris.data
    col_name = ["s_len","s_wid","p_len","p_wid"]
    X      = pd.DataFrame(data, columns=col_name)
    X["Y"] = iris.target
    X["Y"] = X["Y"].apply(lambda y: Y_vectorize(X, y))
    
    #元データ, データ数, 入力層の数, 出力層の数
    return X, X.shape[0], len(col_name), len(X["Y"][0])

def forward(_X, _W_hidden, _W_output):
    target_input_col  = [col for col in range(INPUT_SIZE)]
    target_output_col = [col for col in range(OUTPUT_SIZE)]
    #隠れ層の計算
    #入力層に重みをかけて和をとり、活性化関数をかませる
    list_curt_unit = []
    for curt_unit in range(UNIT_NUM):
        unit_sum = _X[target_input_col].apply(lambda x: np.array(list(x)) * _W_hidden[curt_unit],axis=1)
        unit_sum = unit_sum.sum(1)
        list_curt_unit.append(unit_sum)
    activate = np.tanh(pd.concat(list_curt_unit,axis=1))
    
    #出力層の計算
    #隠れ層の出力に重みをかけて和をとる
    list_curt_unit = []
    for curt_unit in range(OUTPUT_SIZE):
        unit_sum = activate.apply(lambda x: np.array(list(x)) * _W_output[curt_unit],axis=1)
        unit_sum = unit_sum.sum(1)
        list_curt_unit.append(unit_sum)
    output = pd.concat(list_curt_unit,axis=1)
    #発散しないように和を 1 にする
    output["sum"] = output.sum(1)
    output = output.apply(lambda x: x[target_output_col] / x["sum"],axis=1)
    
    #隠れ層の出力, 出力層の出力
    return activate, output

def backpropagation(_X, _activate, _loss, _W_hidden, _W_output):
    #誤差逆伝播
    #隠れ層→出力層の重みを更新する
    W_output_new = []
    for i, col_a in enumerate(_activate):
        for j, col_l in enumerate(_loss):
            W_output_new.append(_W_output[j][i] - LEARNING_RATE * (_activate[col_a] * _loss[col_l]).sum())
    _W_output = np.array(W_output_new).reshape(OUTPUT_SIZE,UNIT_NUM)
    
    #入力層→隠れ層の重みを更新する
    W_hidden_new = []
    for i in range(UNIT_NUM):
        list_loss_backprop = []
        for j, col_l in enumerate(_loss):
            list_loss_backprop.append(_W_output[j][i] * _loss[col_l])
        backprop = (1 - _activate[i] ** 2) * pd.concat(list_loss_backprop,axis=1).sum(1)
        for j, col_x in enumerate(_X):
            if col_x == "Y": continue
            W_hidden_new.append(_W_hidden[i][j] - LEARNING_RATE * (_X[col_x] * backprop).sum())
    _W_hidden = np.array(W_hidden_new).reshape(UNIT_NUM,INPUT_SIZE)
    
    #新しい入力層→隠れ層の重み, 新しい隠れ層→出力層の重み
    return _W_hidden, _W_output
    
def train(_X, _W_hidden, _W_output):
    #訓練
    list_error    = []
    list_accuracy = []
    for epoch in range(0,EPOCH):
        #出力層の結果を計算する
        activate, output = forward(_X, _W_hidden, _W_output)
        
        #誤差を計算し、表示する
        error    = np.absolute(pd.DataFrame(list(_X["Y"])) - output).sum().sum()
        accuracy = output.apply(lambda x: np.argmax(x),axis=1) - pd.DataFrame(list(_X["Y"])).apply(lambda x: np.argmax(x),axis=1)
        for i in range(1,OUTPUT_SIZE): accuracy = np.absolute(accuracy).replace(i,1)
        list_error.append(error)
        list_accuracy.append(DATA_NUM - accuracy.sum())
        print("epoch: "    + str(epoch + 1) + "/" + str(EPOCH) + "  " + \
              "accuracy: " + str(DATA_NUM - accuracy.sum()) + "/" + str(DATA_NUM) + "  " + \
              "error: "    + str(error))
        
        #出力層の評価
        #目的変数との差を計算する
        loss = output - pd.DataFrame(list(X["Y"]))
        #誤差逆伝播
        _W_hidden, _W_output = backpropagation(_X, activate, loss, _W_hidden, _W_output)
    
    #最終的な入力層→隠れ層の重み, 最終的な隠れ層→出力層の重み, 正解数の推移, 誤差の推移
    return _W_hidden, _W_output, list_accuracy, list_error


#########################################################
#irisからデータを生成
X, DATA_NUM, INPUT_SIZE, OUTPUT_SIZE = cre_data_iris()

#重みの初期化
W_hidden = np.random.random_sample((UNIT_NUM, INPUT_SIZE))
W_output = np.random.random_sample((OUTPUT_SIZE, UNIT_NUM))

#訓練
W_hidden, W_output, accuracy, error = train(X, W_hidden, W_output)

#誤差をプロット
pd.DataFrame(accuracy).plot(legend=False)
pd.DataFrame(error).plot(legend=False)

#評価
activate, output = forward(X, W_hidden, W_output)
X["Y"] = output.apply(lambda x: np.argmax(x),axis=1)
X.plot.scatter(x='s_len',y='s_wid',c="Y",cmap='Reds',colorbar=False)
X.plot.scatter(x='p_len',y='p_wid',c="Y",cmap='Reds',colorbar=False)
X.plot.scatter(x='s_len',y='p_wid',c="Y",cmap='Reds',colorbar=False)
X.plot.scatter(x='p_len',y='s_wid',c="Y",cmap='Reds',colorbar=False)
