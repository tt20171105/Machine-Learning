# coding: utf-8
import numpy as np
import activations
from operator  import mul
from functools import reduce

def He(w, inputs_shape):
    return w * np.sqrt(inputs_shape) * np.sqrt(2)
def Xavier(w, inputs_shape):
    return w * np.sqrt(inputs_shape)
def default(w, inputs_shape):
    return w
INITIAL_WEIGHT = {"he":He, "xavier":Xavier, "default":default}

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



class Dense():

    def __init__(self, units_size, activation, inputs_shape=None, initial="he"):
        self.inputs_shape  = inputs_shape
        self.units         = units_size
        self.activation    = activations.get_activation_function(activation)
        self.d_activation  = activations.get_d_activation_function(activation)
        self.initial       = initial
        self.is_last_layer = False

    def build(self):
        if type(self.inputs_shape) is tuple:
            self.inputs_shape = reduce(mul, self.inputs_shape)
        w = np.random.uniform(low=-0.08, high=0.08, size=(self.inputs_shape, self.units))
        self.w = INITIAL_WEIGHT[self.initial](w, self.inputs_shape)
        self.b = np.zeros(self.units)
        self.outputs_shape = self.units

    def forward(self, inputs):
        self.org_shape = inputs.shape
        self.x = inputs.reshape(inputs.shape[0], -1)
        self.z = self.activation(np.dot(self.x, self.w) + self.b)
        return self.z

    def backward(self, inputs):
        if self.is_last_layer:
            self.delta = inputs
        else:
            self.delta = self.d_activation(self.z) * inputs
        self.grad_w    = np.dot(self.x.T, self.delta)
        self.grad_b    = np.dot(np.ones(len(self.x)), self.delta)
        return np.dot(self.delta, self.w.T).reshape(*self.org_shape)

class Conv2D():

    def __init__(self, units_size, kernel_size, activation, inputs_shape=None, stride=1, pad=0):
        self.units  = units_size
        self.kernel = kernel_size
        self.stride = stride
        self.pad    = pad
        self.inputs_shape = inputs_shape
        self.activation   = activations.get_activation_function(activation)
        self.d_activation = activations.get_d_activation_function(activation)

    def build(self):
        self.w = np.random.uniform(low=-0.08, high=0.08, size=(self.units, self.inputs_shape[0], *self.kernel))
        self.b = np.zeros(self.units)
        self.out_H = 1 + int((self.inputs_shape[1] + 2 * self.pad - self.kernel[0]) / self.stride)
        self.out_W = 1 + int((self.inputs_shape[2] + 2 * self.pad - self.kernel[1]) / self.stride)
        self.outputs_shape = (self.units, self.out_H, self.out_W)

    def forward(self, inputs):
        FN, C, FH, FW = self.w.shape
        N,  C, H,  W  = inputs.shape
        self.x     = inputs
        self.col   = im2col(inputs, FH, FW, self.stride, self.pad)
        self.col_w = self.w.reshape(FN, -1).T
        outputs    = self.activation(np.dot(self.col, self.col_w) + self.b)
        outputs    = outputs.reshape(N, self.out_H, self.out_W, -1).transpose(0, 3, 1, 2)
        return outputs

    def backward(self, inputs):
        FN, C, FH, FW = self.w.shape
        deita = inputs.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.grad_b   = np.sum(deita, axis=0)
        self.grad_w   = np.dot(self.col.T, deita)
        self.grad_w   = self.grad_w.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol     = np.dot(deita, self.col_w.T)
        doutputs = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return doutputs

class Pooling():

    def __init__(self, pool_size, stride=1, pad=0, inputs_shape=None):
        self.pool_H = pool_size[0]
        self.pool_W = pool_size[1]
        self.stride = stride
        self.pad    = pad
        self.inputs_shape = inputs_shape
        
    def build(self):
        self.out_H  = int(1 + (self.inputs_shape[1] - self.pool_H) / self.stride)
        self.out_W  = int(1 + (self.inputs_shape[2] - self.pool_W) / self.stride)
        self.outputs_shape = (self.inputs_shape[0], self.out_H, self.out_W)
        
    def forward(self, inputs):
        N, C, H, W = inputs.shape
        col     = im2col(inputs, self.pool_H, self.pool_W, self.stride, self.pad)
        col     = col.reshape(-1, self.pool_H * self.pool_W)
        self.x  = inputs
        self.arg_max = np.argmax(col, axis=1)
        outputs = np.max(col, axis=1)
        outputs = outputs.reshape(N, self.out_H, self.out_W, C).transpose(0, 3, 1, 2)
        return outputs

    def backward(self, inputs):
        delta     = inputs.transpose(0, 2, 3, 1)
        pool_size = self.pool_H * self.pool_W
        dmax      = np.zeros((delta.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
        dmax      = dmax.reshape(delta.shape + (pool_size,))
        dcol      = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        doutputs  = col2im(dcol, self.x.shape, self.pool_H, self.pool_W, self.stride, self.pad)
        return doutputs

class Dropout():

    def __init__(self, drop_ratio):
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.rand(*inputs.shape) > self.drop_ratio
            return inputs * self.mask
        else:
            return inputs * (1.0 - self.drop_ratio)

    def backward(self, inputs):
        return inputs * self.mask
