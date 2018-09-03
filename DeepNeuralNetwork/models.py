
# coding: utf-8
import copy
import numpy as np
from datetime import datetime

import losses
import metrics
from layers import Dense, Dropout

class Sequential():
    
    def __init__(self, seed=None):
        self.history_train = None
        self.history_test  = None
        self.layers     = []
        self.prev_units = None
        if seed is not None:
            np.random.seed(seed)  #シードの固定
        
    def _get_batches(self, x_idx, batch_size, iter_num):
        batch_from = batch_size * iter_num
        batch_to   = batch_size * (iter_num + 1)
        if len(x_idx) - batch_to < batch_size:
            batch_to = len(x_idx)
        return x_idx[batch_from : batch_to]
    
    def add(self, Layer):
        if isinstance(Layer, Dense):
            if Layer.inputs_shape is None:
                Layer.inputs_shape = self.prev_units
            self.prev_units = Layer.units
            Layer.build()
            
        self.layers.append(Layer)
        
    def compile(self, loss, optimizer, metric):
        self.loss        = losses.get_loss_function(loss)
        self.optimizer   = optimizer
        self.metric      = metrics.get_metric_function(metric)
        self.metric_name = metric
        self.layers[-1].is_last_layer = True
    
    def predict(self, x, training=False):
        inputs = x
        for layer in self.layers:
            if isinstance(layer, Dropout):
                inputs = layer.forward(inputs, training)
            else:
                inputs = layer.forward(inputs)
        return inputs
    
    def fit(self, x_train, t_train, x_test, t_test, 
            epochs=100, batch_size=128, verbose=1):
        
        statime = datetime.now()
        
        if self.history_train is None:
            losses_train, results_train = [], []
            losses_test,  results_test  = [], []            
        else:
            losses_train, results_train = self.history_train[0], self.history_train[1]
            losses_test,  results_test  = self.history_test[0],  self.history_test[1]
            
        x_size  = x_train.shape[0]
        iternum = int(len(x_train) / batch_size)
        for epoch in range(epochs):
            total_loss, total_score = 0, 0
            x_idx = np.random.permutation(x_size)
            for i in range(iternum):
                batch_idx = self._get_batches(x_idx, batch_size, i)
                batch_x   = x_train[batch_idx]
                batch_t   = t_train[batch_idx]
                #順伝播
                inputs = self.predict(batch_x, True)
                total_loss  += self.loss(inputs, batch_t)
                total_score += self.metric(inputs, batch_t)
                #誤差逆伝播
                inputs = (inputs - batch_t) / len(batch_idx)
                for layer in reversed(self.layers):
                    inputs = layer.backward(inputs)
                self.optimizer.update_params(self.layers)

            #１エポック分の誤差計算
            #学習
            losses_train.append(total_loss / iternum)
            results_train.append(total_score / iternum)
            #評価
            inputs = self.predict(x_test)
            losses_test.append(self.loss(inputs, t_test))
            results_test.append(self.metric(inputs, t_test))
            #履歴の保存
            self.history_train = [losses_train, results_train]
            self.history_test  = [losses_test,  results_test]

            if verbose==1 and (epoch+1) % 100 == 0:
                print("epoch:%d/%d   train %s:%f loss:%f  test %s:%f loss:%f  time: %s" % 
                      (epoch+1, epochs,
                       self.metric_name, results_train[-1], losses_train[-1],
                       self.metric_name, results_test[-1],  losses_test[-1], datetime.now()-statime))

