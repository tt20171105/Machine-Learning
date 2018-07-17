
# coding: utf-8
import numpy as np

def get_loss_function(loss_function):
    
    #2乗和誤差
    def mean_squared_error(y, t):
        return 0.5 * ((y - t) ** 2).sum()

    #交差エントロピー誤差
    def cross_entropy_error(y, t):
        #発散しないように微小な値を追加する
        delta      = 1e-7
        batch_size = t.shape[0]
        return -(t * np.log(y + delta)).sum() / batch_size
    
    #損失関数の設定
    loss = {"categorical_crossentropy" : cross_entropy_error,
            "mean_squared_error"       : mean_squared_error}
    return loss[loss_function]

