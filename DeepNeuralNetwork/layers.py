
# coding: utf-8
import numpy as np
import activations

class Dense():
    
    def __init__(self, units_size, activation, inputs_shape=None, initial="he"):
        self.inputs_shape  = inputs_shape
        self.units         = units_size
        self.activation    = activations.get_activation_function(activation)
        self.d_activation  = activations.get_d_activation_function(activation)
        self.initial       = initial
        self.is_last_layer = False
        
    def build(self):
        def He(w):
            return w * np.sqrt(self.inputs_shape) * np.sqrt(2)
        def Xavier(w):
            return w * np.sqrt(self.inputs_shape)
        def default(w):
            return w
        initial_weight = {"he"      : He,
                          "xavier"  : Xavier,
                          "default" : default}    
        
        w = np.random.uniform(low=-0.08, high=0.08,
                              size=(self.inputs_shape, self.units))
        self.w = initial_weight[self.initial](w)
        self.b = np.zeros(self.units)
        
    def forward(self, inputs):
        self.x = inputs
        self.z = self.activation(np.dot(inputs, self.w) + self.b)
        return self.z
        
    def backward(self, inputs):
        if self.is_last_layer:
            self.delta = inputs
        else:
            self.delta = self.d_activation(self.z) * inputs
        self.grad_w    = np.dot(self.x.T, self.delta)
        self.grad_b    = np.dot(np.ones(len(self.x)), self.delta)
        return np.dot(self.delta, self.w.T)

class Dropout():

    def __init__(self, drop_ratio):
        self.drop_ratio = drop_ratio
        self.mask       = None
    
    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.rand(*inputs.shape) > self.drop_ratio
            return inputs * self.mask
        else:
            return inputs * (1.0 - self.drop_ratio)

    def backward(self, inputs):
        return inputs * self.mask

