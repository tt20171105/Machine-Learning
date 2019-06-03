
# coding: utf-8
import os, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets

def iris():
    iris = datasets.load_iris()
    return iris.data

class Kmeans():
    
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        
    def _calc_center(self, cluster_num):
        # 重心を計算
        return self.data[np.where(self.cluster==cluster_num)].mean(axis=0)
        
    def _calc_distance(self, center):
        # ユークリッド距離を計算
        return ((self.data - center)**2).sum(axis=1)
        
    def fit(self, data):
        self.data    = data
        # 初期クラスタの決定
        self.cluster = np.array([random.randint(0, self.k) for _ in range(len(data))])
        distances    = np.zeros((data.shape[0], self.k))
        for _ in range(self.max_iter):
            for i in range(self.k):
                # 現在のクラスタの重心を計算
                center   = self._calc_center(i)
                # 全てのデータとの距離を計算
                distance = self._calc_distance(center)
                distances[:, i] = distance
            cluster = distances.argmin(axis=1)
            # 収束判定(クラスタが前回と変わっていなければ収束)
            if np.abs(self.cluster - cluster).sum()==0:
                break
            self.cluster = cluster
        else:
            print("既定のループ回数で収束しませんでした。")

    def result_show(self, xdim=0, ydim=1):
        for i in range(self.k):
            x = self.data[np.where(self.cluster==i), xdim]
            y = self.data[np.where(self.cluster==i), ydim]
            plt.scatter(x=x, y=y, color=cm.hsv(i/k))

data = iris()

k = 2
kmeans = Kmeans(k)
kmeans.fit(data)
kmeans.result_show()

