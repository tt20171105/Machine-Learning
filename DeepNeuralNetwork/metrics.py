
# coding: utf-8
import numpy as np

def get_metric_function(metrics_function):
    #正解率
    def accuracy(y, t):
        batch_size = t.shape[0]
        acc = np.argmax(y, axis=1) - np.argmax(t, axis=1)
        acc = np.where(acc==0, 1, 0).sum()
        return acc / batch_size

    #RMSE
    def root_mean_square_error(y, t):
        rmse = np.sqrt(np.mean((t - y) ** 2))
        return rmse

    #MSE
    def mean_square_error(y, t):
        mse = np.mean((t - y) ** 2)
        return mse
    
    #評価関数の設定
    metrics = {"accuracy" : accuracy,
               "rmse"     : root_mean_square_error,
               "mse"      : mean_square_error}
    return metrics[metrics_function]

