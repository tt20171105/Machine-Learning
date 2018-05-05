# coding: utf-8
"""
Created on Sun Sep 10 22:02:06 2017

@author: tt20171105
"""
import itertools
import numpy as np
import pandas as pd
from sklearn import datasets

#データの準備(iris)
def cre_data_iris(is_vectorize=True):
    #目的変数をベクトル化
    def t_vectorize(t, sorted_t):
        vector_t = []
        for _t in sorted_t:
            if _t==t: 
                vector_t.append(1)
            else:
                vector_t.append(0)
        return vector_t
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data,   columns=["s_len","s_wid","p_len","p_wid"])
    t = pd.DataFrame(iris.target, columns=["t"])
    output_layer_size = 1
    if is_vectorize:
        sorted_t = sorted(t.t.unique())
        t["t"]   = t.apply(lambda t: t_vectorize(t.values[0], sorted_t), axis=1)
        output_layer_size = len(t.t[0])
    #元データ, 入力層の数, 正解ラベル, 出力層の数
    return x, x.shape[1], t, output_layer_size

class DeepNeuralNetwork():
    
    def __init__(self, seed=None):
        self.unit_size = []
        self.h         = []
        self.dh        = []
        self.loss      = None
        #学習履歴の初期化
        self.history   = None
        #シード
        self.seed      = seed
    
    def _initial_weight(self):
        #シードの固定
        if not self.seed is None:
            np.random.seed(self.seed)

        list_w = []
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
    
    #恒等関数
    def _identify(self, x):
        return x
    
    #ソフトマックス関数
    def _softmax(self, x):
        max_x     = np.max(x)
        exp_x     = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x)
        y         = exp_x / sum_exp_x
        return y 
    
    ##############################
    #損失関数
    #2乗和誤差
    def _mean_squared_error(self, df_y, df_t):
        return 0.5 * ((df_y - df_t) ** 2).sum().sum()
    
    #交差エントロピー誤差
    def _cross_entropy_error(self, df_y, df_t):
        #発散しないように微小な値を追加する
        delta = 1e-7
        return - (df_t * np.log(df_y + delta)).sum().sum()

    ##############################
    #評価関数
    #正解率
    def _accuracy(self, df_y, df_t, data_num):
        l   = lambda x: x.values.argmax()
        acc = df_y.apply(l, axis=1) - df_t.apply(l, axis=1)
        acc = list(acc).count(0) / data_num
        return acc
        
    #RMSE
    def _rmse(self, df_y, df_t, data_num):
        rmse = np.sqrt(np.mean((df_t - df_y) ** 2))
        return rmse
    
    ##############################
    #順伝播
    def _forward(self, x, prd=False):
        
        def unit_sum(df, unit, w, b):
            list_unit = []
            for curt in range(unit):
                list_unit.append((df * w[curt]).sum(1) + b[curt])
            return pd.concat(list_unit, axis=1)

        def activation(df, h):
            if h.__name__=="_softmax":
                return df.apply(lambda x: h(x), axis=1)
            return df.applymap(h)
        
        activated, dactivated, df = [], [], x
        for i in range(len(self.unit_size)):
            #隠れ層の計算
            df = unit_sum(df, self.unit_size[i], self.hid_w[i], self.hid_b[i])
            activated.append(activation(df, self.h[i]))
            dactivated.append(activation(df, self.dh[i]))
        else:
            #最後の隠れ層→出力層の計算
            df = unit_sum(df, self.out_layer_size, self.hid_w[i+1], self.hid_b[i+1])
            activated.append(activation(df, self.h[i+1]))
        
        #活性化関数計算後
        if prd:
            return activated[self.hid_layer_num]
        else:
            self.activated  = activated
            self.dactivated = dactivated
        
    ##############################
    #誤差逆伝播
    def _backward(self, x, alpha):
        
        def grad(tg_df, tg_w, tg_b, bf_df):
            w = []
            #勾配を求める
            iter_col = itertools.product(tg_df.columns, bf_df.columns)
            for idx, cols in enumerate(iter_col):
                grad = (tg_df[cols[0]] * bf_df[cols[1]]).sum()
                w.append(tg_w[0][idx] - alpha * grad)
            grad = (tg_df * tg_b).sum()
            b    = tg_b - alpha * grad
            return w, b

        w_new, b_new = [], []
        #隠れ層→出力層の重みを更新する
        w, b = grad(self.y_t,
                    self.hid_w[self.hid_layer_num].reshape(1,-1),
                    self.hid_b[self.hid_layer_num],
                    self.activated[self.hid_layer_num-1])
        #更新後の重みとバイアス
        w_new.append(np.array(w).reshape(self.out_layer_size,
                                         self.unit_size[self.hid_layer_num-1]))
        b_new.append(b)
        
        #入力層→隠れ層、隠れ層→隠れ層の重みを更新する
        for i in range(1, self.hid_layer_num+1):
            hid     = []
            next_df = self.y_t if i==1 else self.activated[self.hid_layer_num-i+1]
            next_w  = self.hid_w[self.hid_layer_num-i+1]
            for j in range(next_w.shape[1]):
                hid.append((next_df * next_w[:,j]).sum(1))
            hid  = self.dactivated[self.hid_layer_num-i] * pd.concat(hid, axis=1)
            
            before_df   = x if i==self.hid_layer_num else self.activated[self.hid_layer_num-i-1]
            before_unit = self.in_layer_size if i==self.hid_layer_num else self.unit_size[self.hid_layer_num-i-1]
            w, b = grad(hid,
                        self.hid_w[self.hid_layer_num-i].reshape(1,-1),
                        self.hid_b[self.hid_layer_num-i],
                        before_df)
            #更新後の重みとバイアス
            w_new.append(np.array(w).reshape(self.unit_size[self.hid_layer_num-i],
                                             before_unit))
            b_new.append(b)
        
        w_new.reverse()
        b_new.reverse()
        self.hid_w = w_new
        self.hid_b = b_new
        
    ##############################
    #層の設定
    def add_input_layer(self, layer_size, std=False):
        #層の設定
        self.in_layer_size = layer_size
        #データの変換処理
        self.std           = std
    
    def add_hidden_layer(self, layer_size, activation):
        #層の設定
        self.unit_size.append(layer_size)
        self.hid_layer_num = len(self.unit_size)
        
        #活性化関数の設定
        if   activation=="ReLU":
            self.h.append(self._ReLU)
            self.dh.append(self._dReLU)
        elif activation=="tanh":
            self.h.append(self._tanh)
            self.dh.append(self._dtanh)
        elif activation=="sigmoid":
            self.h.append(self._sigmoid)
            self.dh.append(self._dsigmoid)
        else:
            print("we don't support activation function '{0}'. \nBecause we use ReLU.".format(activation))
            self.h.append(self._ReLU)
            self.dh.append(self._dReLU)
            
    def add_output_layer(self, layer_size, activation, loss, evaluation):
        #層の設定
        self.out_layer_size = layer_size
        
        #活性化関数の設定
        if   activation=="softmax":
            self.h.append(self._softmax)
        elif activation=="identify":
            self.h.append(self._identify)
        else:
            print("we don't support activation function '{0}'. \nBecause we use softmax.".format(activation))
            self.h.append(self._softmax)            
            
        #損関数の設定 
        if   loss=="categorical_crossentropy":
            self.loss = self._cross_entropy_error
        elif loss=="mean_squared_error":
            self.loss = self._mean_squared_error
        else:
            print("we don't support loss function '{0}'. \nBecause we use categorical crossentropy.".format(loss))
            self.loss = self._cross_entropy_error
            
        #評価関数の設定
        if   evaluation=="accuracy":
            self.eval_name  = "acc"
            self.evaluation = self._accuracy
        elif evaluation=="rmse":
            self.eval_name  = "rmse"
            self.evaluation = self._rmse            
        else:
            print("we don't support evaluation function '{0}'. \nBecause we use accuracy.".format(evaluation))
            self.eval_name  = "acc"
            self.evaluation = self._accuracy            
    
    #学習
    def fit(self, x, t, epochs=100, alpha=0.0000001, verbose=1):
        
        if self.history is None:
            self._initial_weight()
            list_loss = []
            list_eval = []
        else:
            list_loss = self.history[0]
            list_eval = self.history[1]
            
        if self.std:
            x = x.apply(lambda x: (x - x.mean()) / x.std())
            
        #訓練
        for epoch in range(epochs):
            #出力層の結果を計算する
            self._forward(x)
            
            #誤差を計算し、表示する
            df_y = self.activated[self.hid_layer_num]
            df_t = pd.DataFrame(list(t.iloc[:,0]))
            loss = self.loss(df_y, df_t)
            list_loss.append(loss)
            data_num   = x.shape[0]
            evaluation = self.evaluation(df_y, df_t, data_num)
            list_eval.append(evaluation)
            if verbose==1:
                print("epoch: %d / %d   %s: %f   loss: %f" % (epoch + 1, epochs, self.eval_name, evaluation, loss))
        
            #誤差逆伝播
            self.y_t = df_y - df_t
            self._backward(x, alpha)
            #履歴の保存
            self.history = [list_loss, list_eval]
    
    ##############################
    #予測
    def predict(self, x):
        if self.std:
            x = x.apply(lambda x: (x - x.mean()) / x.std())
        #出力層の結果を計算する
        df = self._forward(x, prd=True)
        if self.eval_name=="acc":
            return df.apply(lambda x: x.values.argmax(), axis=1)
        else:
            return df

#########################################################
#irisからデータを生成
X, INPUT_SIZE, T, OUTPUT_SIZE = cre_data_iris()

dnn = DeepNeuralNetwork(seed=15)
dnn.add_input_layer(layer_size=INPUT_SIZE, std=False)
dnn.add_hidden_layer(layer_size=4, activation="ReLU")
dnn.add_hidden_layer(layer_size=4, activation="ReLU")
dnn.add_hidden_layer(layer_size=4, activation="ReLU")
dnn.add_output_layer(layer_size=OUTPUT_SIZE, activation="softmax", loss="categorical_crossentropy", evaluation="accuracy")

EPOCHS        = 200     #エポック数
ALPHA         = 0.00001  #学習率
dnn.fit(X, T, epochs=EPOCHS, alpha=ALPHA, verbose=1)

#誤差をプロット
get_ipython().run_line_magic('matplotlib', 'inline')

pd.DataFrame(dnn.history[0]).plot(legend=False, title="loss")
pd.DataFrame(dnn.history[1]).plot(legend=False, title=dnn.eval_name)

#評価
X["Y"] = dnn.predict(X)
colors = ["red","blue","green"]
colors = [colors[i] for i in X.Y]
X.plot.scatter(x='p_len',y='s_wid',c=colors)
X.plot.scatter(x='p_len',y='p_wid',c=colors)
X.plot.scatter(x='s_len',y='p_wid',c=colors)
X.plot.scatter(x='p_len',y='s_wid',c=colors)
X.drop("Y", axis=1, inplace=True)

