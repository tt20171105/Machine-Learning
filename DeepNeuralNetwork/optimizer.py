
# coding: utf-8
import numpy as np
from layers import Dropout

def get_update_layers(layers):
    update_layers = {}
    for idx, layer in enumerate(layers):
        if not isinstance(layer, Dropout):
            update_layers[idx] = layer
    return update_layers

class SGD():
    
    def __init__(self, lr=0.01):
        self.lr = 0.01
        
    def update_params(self, layers):
        update_layers = get_update_layers(layers)
        for idx, layer in update_layers.items():
            layers[idx].w -= self.lr * layer.grad_w
            layers[idx].b -= self.lr * layer.grad_b

class Momentum():
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr       = lr
        self.momentum = momentum
        self.v        = {}
        
    def update_params(self, layers):
        update_layers = get_update_layers(layers)
        if self.v == {}:
            for idx, update_layer in update_layers.items():
                self.v[idx] = {"w" : np.zeros_like(update_layer.w),
                               "b" : np.zeros_like(update_layer.b)}
                
        for idx, layer in update_layers.items():
            self.v[idx]["w"] = self.momentum * self.v[idx]["w"] - self.lr * layer.grad_w
            self.v[idx]["b"] = self.momentum * self.v[idx]["b"] - self.lr * layer.grad_b
            layers[idx].w   += self.v[idx]["w"]
            layers[idx].b   += self.v[idx]["b"]

class AdaGrad():
    
    def __init__(self, lr=0.01, epsilon=1e-7):
        self.lr      = lr
        self.epsilon = epsilon
        self.h       = {}
        
    def update_params(self, layers):
        update_layers = get_update_layers(layers)
        if self.h == {}:
            for idx, update_layer in update_layers.items():
                self.h[idx] = {"w" : np.zeros_like(update_layer.w),
                               "b" : np.zeros_like(update_layer.b)}
        
        for idx, layer in update_layers.items():
            self.h[idx]["w"] += layer.grad_w * layer.grad_w
            self.h[idx]["b"] += layer.grad_b * layer.grad_b
            layers[idx].w    -= self.lr * layer.grad_w / (np.sqrt(self.h[idx]["w"]) + self.epsilon)
            layers[idx].b    -= self.lr * layer.grad_b / (np.sqrt(self.h[idx]["b"]) + self.epsilon)

class RMSprop():

    def __init__(self, lr=0.01, decay=0.99, epsilon=1e-7):
        self.lr      = lr
        self.decay   = decay
        self.epsilon = epsilon
        self.h       = {}
        
    def update_params(self, layers):
        update_layers = get_update_layers(layers)
        if self.h == {}:
            for idx, update_layer in update_layers.items():
                self.h[idx] = {"w" : np.zeros_like(update_layer.w),
                               "b" : np.zeros_like(update_layer.b)}
        
        for idx, layer in update_layers.items():
            self.h[idx]["w"] *= self.decay
            self.h[idx]["b"] *= self.decay
            self.h[idx]["w"] += (1 - self.decay) * layer.grad_w * layer.grad_w
            self.h[idx]["b"] += (1 - self.decay) * layer.grad_b * layer.grad_b
            layers[idx].w    -= self.lr * layer.grad_w / (np.sqrt(self.h[idx]["w"]) + self.epsilon)
            layers[idx].b    -= self.lr * layer.grad_b / (np.sqrt(self.h[idx]["b"]) + self.epsilon)

class Adam():
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.iter    = 0
        self.m       = {}
        self.v       = {}
        
    def update_params(self, layers):
        update_layers = get_update_layers(layers)
        if self.m == {}:
            for idx, update_layer in update_layers.items():
                self.m[idx] = {"w" : np.zeros_like(update_layer.w),
                               "b" : np.zeros_like(update_layer.b)}
                self.v[idx] = {"w" : np.zeros_like(update_layer.w),
                               "b" : np.zeros_like(update_layer.b)}
                
        self.iter += 1
        lr_t       = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for idx, layer in update_layers.items():
            self.m[idx]["w"] += (1 - self.beta1) * (layer.grad_w      - self.m[idx]["w"])
            self.m[idx]["b"] += (1 - self.beta1) * (layer.grad_b      - self.m[idx]["b"])
            self.v[idx]["w"] += (1 - self.beta2) * (layer.grad_w ** 2 - self.v[idx]["w"])
            self.v[idx]["b"] += (1 - self.beta2) * (layer.grad_b ** 2 - self.v[idx]["b"])
            layers[idx].w    -= lr_t * self.m[idx]["w"] / (np.sqrt(self.v[idx]["w"]) + self.epsilon)
            layers[idx].b    -= lr_t * self.m[idx]["b"] / (np.sqrt(self.v[idx]["b"]) + self.epsilon)

