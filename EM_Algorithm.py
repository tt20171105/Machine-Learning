
# coding: utf-8
import numpy as np
from numpy.random import uniform
from numpy.linalg import det, inv
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def iris():
    iris = datasets.load_iris()
    return iris.data

# EMアルゴリズム
class EM_Algorithm():
    
    def __init__(self, k, max_iter=100, seed=15):
        self.k = k
        self.max_iter = max_iter
        self.seed     = seed
        np.random.seed(seed)
        
    def gauss(self, data):
        """ 逆行列を計算
　        　クラスタ1   クラスタ2
  　    　[[[逆行列1,    逆行列1],
      　  　[逆行列2,    逆行列2]],
         　[[逆行列3,    逆行列3],
        　  [逆行列4 ,   逆行列4]]]
        """
        invs = inv(self.covs.T).T
        """ 偏差（各データと平均の差）を計算
　           クラスタ1  クラスタ2
  　      [[[変数1差分, 変数1差分],  1行目データ
    　      [変数2差分, 変数2差分],
      　    [  ・・・ ,   ・・・ ],
        　 [[変数1差分, 変数1差分],  2行目データ
         　 [  ・・・ ,   ・・・ ]]]
        """
        devs = data[:, :, None] - self.means
        """ 差分と分散共分散行列の内積を計算
        　1行目のデータについて
　                  [1, 2, 3]     クラスタ1   　 クラスタ2
  　      [A, B ,C]*[4, 5, 6]=[A*1+A*4+A*7, ・・],[・・]
    　              [7, 8, 9]
      　  これを縦にもちたいため、einsumを使い計算結果は以下となる。
        　   クラスタ1  クラスタ2
　        [[[変数1内積, 変数1内積],  1行目データ
  　        [変数2内積, 変数2内積],  
    　      [  ・・・ ,   ・・・ ]],
      　   [[変数1内積, 変数1内積],  2行目データ
        　  [  ・・・ ,   ・・・ ]]]
        """
        exps = np.einsum('nik,ijk->njk', devs, invs)
        """ 内積と差分を掛け、クラスタと各データごとに和を計算する
                  　クラスタ1            　　　　　　クラスタ2
        [[sum(変数1～3内積×変数1～3差分), sum(変数1～3内積×変数1～3差分)],  1行目データ
         [sum(変数1～3内積×変数1～3差分), sum(変数1～3内積×変数1～3差分)],  2行目データ
         [ ・・・ ]]
        """
        exps = np.sum(exps * devs, axis=1)
        """ 多次元正規分布
　        　現在の平均、分散における、各データの生成確率
　               クラスタ１              クラスタ２
  　      [[1行目データの生成確率, 1行目データの生成確率],
    　     [2行目データの生成確率, 2行目データの生成確率],
      　   [ ・・・ ]]
        """
        return np.exp(-0.5 * exps) / np.sqrt(det(self.covs.T).T * (2 * np.pi) ** self.ndim)
    
    def _Estep(self):
        """ 各データの生成確率に混合比をかけて負担率を計算
        　　計算した負担率を0～1に正規化する
        　　※混合比：各クラスタが全データの何割を負担しているか
          　※負担率：各データがどのクラスタから生成されたかの確率
         　クラスタ1  クラスタ2
        　[[負担率1,   負担率2],  1行目のデータ
         　[負担率1,   負担率2],　2行目のデータ
         　[・・・ ,   ・・・ ]]
         """
        resps  = self.weights * self.gauss(self.data)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _Mstep(self, resps):
        """ クラスタ別の負担率の和を計算
    　　　　計算結果を0～1に正規化したものを新たな混合比とする
　        　　※負担率も0～1に正規化されているので、
  　        　　負担率を行ごとの和は1になる
    　        　そのためsum(Nk)はデータ数(len(self.data))と一致する
        　[クラスタ1の負担率合計, クラスタ2の負担率合計, ・・・]
        """
        Nk = np.sum(resps, axis=0)
        self.weights = Nk / len(self.data)
        """ 以下の内積を計算し、負担率の総和で割ったものを新たな平均とする
    　　　　（普通の平均の計算時に負担率を掛ける）
        　self.data.T  (dim×n)
        　[[1行目データの変数1, 2行目データの変数1, ・・],
           [1行目データの変数2, 2行目データの変数2, ・・],
           [ ・・・ ],]
          reps　(n×k)
          [[1行目データのクラスタ1負担率, 1行目データのクラスタ2負担率],
           [2行目データのクラスタ1負担率, 2行目データのクラスタ2負担率],
           [ ・・・ ]]
        """
        self.means = self.data.T.dot(resps) / Nk
        """ 偏差に負担率を掛け、さらに偏差を掛ける
        　　計算結果を負担率の総和で割ったものを新たな分散共分散とする
          　（普通の分散共分散の計算時に負担率を掛ける）
        """
        devs      = self.data[:, :, None] - self.means
        self.covs = np.einsum('nik,njk->ijk', devs, devs * np.expand_dims(resps, 1)) / Nk

    def _convergence(self):
        return np.hstack((self.weights.ravel(),
                          self.means.ravel(),
                          self.covs.ravel()))
    
    def fit(self, data):
        self.data = data
        # 次元数
        self.ndim = np.size(data, 1)
        # 混合比
        # [[クラスタ1, クラスタ2, …]]
        self.weights = np.ones(self.k) / self.k
        # 平均
        self.means   = uniform(data.min(), data.max(), (self.ndim, self.k))
        # 分散共分散
        self.covs    = np.repeat(10 * np.eye(self.ndim), self.k).reshape(self.ndim, self.ndim, self.k)
        
        for _ in range(self.max_iter):
            # 収束判定用（パラメータ更新前）
            params = self._convergence()
            # Eステップ
            resps  = self._Estep()
            # Mステップ
            self._Mstep(resps)
            # 収束判定
            if np.allclose(params, self._convergence()):
                break
        else:
            print("既定のループ回数で収束しませんでした。")

    def predict_prob(self, data):
        predicted = self.weights * self.gauss(data)
        return predicted

    def classify(self, data):
        predicted = self.weights * self.gauss(data)
        return np.argmax(predicted, axis=1)

    def result_show(self, xdim=0, ydim=1):
        cluster = self.classify(self.data)
        for i in range(self.k):
            x = self.data[np.where(cluster==i), xdim]
            y = self.data[np.where(cluster==i), ydim]
            plt.scatter(x=x, y=y, color=cm.hsv(i/self.k))

data = iris()

K  = 2
em = EM_Algorithm(K)
em.fit(data)

em.result_show()

