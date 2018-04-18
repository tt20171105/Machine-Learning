# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:02:06 2017

@author: tt20171105
"""
import numpy as np
import pandas as pd
from sklearn import datasets

#データの準備(iris)
def cre_data_iris():
    #目的変数をベクトル化
    def Y_vectorize(y, sorted_y):
        vector_y = []
        for _y in sorted_y:
            if _y==y: 
                vector_y.append(1)
            else:
                vector_y.append(0)
        return vector_y
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data,   columns=["s_len","s_wid","p_len","p_wid"])
    Y = pd.DataFrame(iris.target, columns=["Y"])
    sorted_y = sorted(Y.Y.unique())
    Y["Y"]   = Y.apply(lambda y: Y_vectorize(y.values[0], sorted_y), axis=1)
    #元データ, データ数, 入力層の数, 正解ラベル, 出力層の数
    return X, X.shape[0], X.shape[1], Y, len(Y.Y[0])

class DeepNeuralNetwork():
    
    def __init__(self,
                 in_layer_size,
                 out_layer_size,
                 list_unit_size,
                 activation="ReLU",
                 seed=""):
        self.in_layer_size  = in_layer_size
        self.hid_layer_num  = len(list_unit_size)
        self.out_layer_size = out_layer_size
        self.unit_size      = list_unit_size
        #重みの初期化
        self._initial_weight(seed)
        #学習履歴の初期化
        self.history = None
        #活性化関数の設定
        if   activation=="ReLU":
            self.h  = self._ReLU
            self.dh = self._dReLU
        elif activation=="tanh":
            self.h  = self._tanh
            self.dh = self._dtanh
        elif activation=="sigmoid":
            self.h  = self._sigmoid
            self.dh = self._dsigmoid
        else:
            print("we don't support activation function '{0}'. \nBecause we use ReLU.".format(activation))
            self.h  = self._ReLU
            self.dh = self._dReLU

    def _initial_weight(self, seed):
        list_w = []
        #シードの固定
        if seed!="": np.random.seed(seed)
        #入力層→1つ目の隠れ層の重み
        w = np.random.random_sample((self.unit_size[0], self.in_layer_size))
        list_w.append(w)
        #次の隠れ層の重み
        if self.hid_layer_num > 1:
            for l in range(self.hid_layer_num-1):
                w = np.random.random_sample((self.unit_size[l+1], self.unit_size[l]))
                list_w.append(w)
        #最後の隠れ層→出力層の重み
        w = np.random.random_sample((self.out_layer_size, self.unit_size[self.hid_layer_num-1]))
        list_w.append(w)
        #バイアス
        list_b = []
        for i in range(len(list_w)):
            list_b.append(np.ones(list_w[i].shape[0]))
        
        self.hid_w = list_w
        self.hid_b = list_b
    
    ##############################
    #活性化関数
    #シグモイド関数
    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-x))
    def _dsigmoid(self, x):
        return x * (1. - x)
    #双曲線正接関数
    def _tanh(self, x):
        return np.tanh(x)
    def _dtanh(self, x):
        return 1. - x * x
    #ランプ関数
    def _ReLU(self, x):
        return x * (x > 0)
    def _dReLU(self, x):
        return 1. * (x > 0)
    #ソフトマックス関数
    def _softmax(self, x):
        max_x     = np.max(x)
        exp_x     = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x)
        y         = exp_x / sum_exp_x
        return y 
    
    ##############################
    #順伝播
    def _forward(self, x, prd=False):
        
        def unit_sum(df, unit, w, b):
            list_unit = []
            for curt in range(unit):
                x_w = df.apply(lambda x: np.array(x) * w[curt], axis=1)
                list_unit.append(x_w.sum(1) + b[curt])
            return pd.concat(list_unit, axis=1)

        def activation(df, h):
            if h.__name__=="_softmax":
                return df.apply(lambda x: h(x), axis=1)
            return df.applymap(h)
        
        activated, dactivated, df = [], [], x
        for i in range(len(self.unit_size)):
            #隠れ層の計算
            df = unit_sum(df, self.unit_size[i], self.hid_w[i], self.hid_b[i])
            activated.append(activation(df, self.h))
            dactivated.append(activation(df, self.dh))
        else:
            #最後の隠れ層→出力層の計算
            df = unit_sum(df, self.out_layer_size, self.hid_w[i+1], self.hid_b[i+1])
            self.test = df
            activated.append(activation(df, self._softmax))
        
        #活性化関数計算後
        if prd:
            return activated[self.hid_layer_num]
        else:
            self.activated  = activated
            self.dactivated = dactivated
        
    ##############################
    #誤差逆伝播
    def _backward(self, x, alpha):
        w_new, b_new = [], []
        #隠れ層→出力層の重みを更新する
        w, b      = [], []
        target_w  = self.hid_w[self.hid_layer_num]
        target_b  = self.hid_b[self.hid_layer_num]
        before_df = self.activated[self.hid_layer_num-1]
        for i, c_l in enumerate(self.loss):
            for j, c_h in enumerate(before_df):
                #重みの更新
                _sum = (self.loss[c_l] * before_df[c_h]).sum()
                w.append(target_w[i][j] - alpha * _sum)
            #バイアスの更新
            _sum = (self.loss[c_l] * target_b[i]).sum()
            b.append(target_b[i] - alpha * _sum)
        #更新後の重みとバイアスを退避
        w_new.append(np.array(w).reshape(self.out_layer_size,
                                         self.unit_size[self.hid_layer_num-1]))
        b_new.append(b)
        
        #入力層→隠れ層、隠れ層→隠れ層の重みを更新する
        def calc_hid_backward(target_df, target_w, target_b, target_unit,
                              before_df, before_unit, next_df, next_w):
            w, b = [], []
            for i, c_d in enumerate(target_df):
                hid = []
                for j, c_n in enumerate(next_df):
                    hid.append(next_w[j][i] * next_df[c_n])
                hid = target_df[c_d] * (pd.concat(hid, axis=1).sum(1))
                for j, c_t in enumerate(before_df):
                    #重みの更新
                    _sum = (hid * before_df[c_t]).sum()
                    w.append(target_w[i][j] - alpha * _sum)
                #バイアスの更新
                _sum = (hid * target_b[i]).sum()
                b.append(target_b[i] - alpha * _sum)
            return np.array(w).reshape(target_unit, before_unit), b
        
        for i in range(1, self.hid_layer_num+1):
            target_df   = self.dactivated[self.hid_layer_num-i]
            target_w    = self.hid_w[self.hid_layer_num-i]
            target_b    = self.hid_b[self.hid_layer_num-i]
            target_unit = self.unit_size[self.hid_layer_num-i]
            if i==self.hid_layer_num:
                before_df   = x
                before_unit = self.in_layer_size
            else:
                before_df   = self.activated[self.hid_layer_num-i-1]
                before_unit = self.unit_size[self.hid_layer_num-i-1]
            if i==1:
                next_df = self.loss
            else:
                next_df = self.activated[self.hid_layer_num-i+1]
            next_w      = self.hid_w[self.hid_layer_num-i+1]
            
            w, b = calc_hid_backward(target_df, target_w, target_b, target_unit,
                                     before_df, before_unit, next_df, next_w)
            w_new.append(w)
            b_new.append(b)
        
        w_new.reverse()
        b_new.reverse()
        self.hid_w = w_new
        self.hid_b = b_new
        
    ##############################
    #学習
    def fit(self, x, y, epochs=100, alpha=0.0000001, verbose=1):
        #学習履歴のロード
        if self.history is None:
            list_loss = []
            list_acc  = []
        else:
            list_loss = self.history[0]
            list_acc  = self.history[1]
        #訓練
        for epoch in range(epochs):
            #出力層の結果を計算する
            self._forward(x)
            
            #誤差を計算し、表示する
            df_out_layer = self.activated[self.hid_layer_num]
            df_y         = pd.DataFrame(list(y.iloc[:,0]))
            self.loss    = df_out_layer - df_y
            loss = np.absolute(self.loss).sum().sum()
            l    = lambda x: x.values.argmax()
            acc  = df_out_layer.apply(l, axis=1) - df_y.apply(l, axis=1)
            acc  = list(acc).count(0) / x.shape[0]
            list_loss.append(loss)
            list_acc.append(acc)
            if verbose==1:
                print("epoch: {0} / {1}   acc: {2}   loss: {3}".format(
                      str(epoch + 1), str(epochs), str(round(acc,2)), str(loss)))
        
            #誤差逆伝播
            self._backward(x, alpha)
            #履歴の保存
            self.history = [list_loss, list_acc]
    
    ##############################
    #予測
    def predict(self, x):
        #出力層の結果を計算する
        df = self._forward(x, prd=True)
        return df.apply(lambda x: x.values.argmax(), axis=1)


#########################################################
#ハイパーパラメータ
LIST_UNIT_NUM = [4,4]    #隠れ層のユニット数
EPOCHS        = 1000     #エポック数
ALPHA         = 0.00001  #学習率

#irisからデータを生成
X, DATA_NUM, INPUT_SIZE, Y, OUTPUT_SIZE = cre_data_iris()

dnn = DeepNeuralNetwork(INPUT_SIZE, OUTPUT_SIZE, LIST_UNIT_NUM, seed=15)
dnn.fit(X, Y, epochs=EPOCHS, alpha=ALPHA)

#誤差をプロット
pd.DataFrame(dnn.history[0]).plot(legend=False)
pd.DataFrame(dnn.history[1]).plot(legend=False)

#評価
X["Y"] = dnn.predict(X)
X.plot.scatter(x='s_len',y='s_wid',c="Y",cmap='Reds',colorbar=False)
X.plot.scatter(x='p_len',y='p_wid',c="Y",cmap='Reds',colorbar=False)
X.plot.scatter(x='s_len',y='p_wid',c="Y",cmap='Reds',colorbar=False)
X.plot.scatter(x='p_len',y='s_wid',c="Y",cmap='Reds',colorbar=False)
