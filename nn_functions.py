import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x.shape)
    grad[x>=0] = 1
    return grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def numerical_gradient(f, x):
    h = 1e-1
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return grad


def cross_entropy_error(y, t):
    
    delta = 1e-7
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    qq = y[np.arange(batch_size), t]        
    ll = np.log(qq+delta)
    ww = -np.sum(ll)
    return ww / batch_size