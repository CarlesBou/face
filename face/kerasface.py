# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:13:33 2024

@author: Carles
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dropout

from keras.engine.input_layer import InputLayer


def face_contrib(x_sample, model, return_weighted=True):
    if isinstance(x_sample, (pd.core.series.Series)):
        x_sample = x_sample.to_numpy()
        
    W_list, I_vect_list = run_layers(x_sample, model)
    
    for I_index, I_vect in enumerate(I_vect_list):
        if I_index == 0:
            contrib = I_vect[:, None] * W_list[I_index]
        else:
            contrib = (I_vect[:, None] * W_list[I_index]) @ contrib

    if return_weighted:
        contrib[:,1:] = contrib[:,1:] * x_sample
        
    return contrib[1:]


def run_layers(layer_input, model, return_PI_list=False):
    W_list = []
    I_vect_list = []
    
    x_ext = np.hstack((1, layer_input))
    
    for layer in model.layers:
        if isinstance(layer, (InputLayer, Dropout)):
            continue
        
        if len(layer.get_weights()) == 2:
            bias = layer.get_weights()[1]
        else:
            bias = None
            
        w = layer.get_weights()[0]    
        w_T_ext = get_transposed_ext(w, bias)
        
        W_list.append(w_T_ext)
        
        H, I_vect = get_I_activation(layer.activation, x_ext, w_T_ext, bias)
        
        I_vect_list.append(I_vect)
        
        x_ext = H
          
    return W_list, I_vect_list


'''
Obtenemos la matriz pseudoidentidad para una función linean para una entrada
de la capa (que tiene unos pesos y bias asodicados, puede que bias=0)
'''
def get_I_activation(activation, x, w=None, bias=None):
  if isinstance(activation, tf.keras.layers.ReLU):
    alpha = activation.negative_slope
    return get_I_relu(x, w, bias, alpha)
  elif isinstance(activation, tf.keras.layers.LeakyReLU):
    alpha = activation.alpha
    return get_I_relu(x, w, bias, alpha)
  else:
    match activation.__name__:
      case 'relu':
        return get_I_relu(x, w, bias, 0)
      case 'linear':
        return get_I_linear(x, w, bias)      
      case 'hard_sigmoid':
        return get_I_hard_sigmoid(x, w, bias)
      case 'hard_tanh':
        return get_I_hard_tanh(x, w, bias)
      case _:   
        assert(f'Unsupported activation function \'{activation.__name__}\'')
    


'''
Obtenemos la matriz pseudoidentidad para una ReLU para una entrada
de la capa (que tiene unos pesos y bias asodicados)
'''

def get_I_linear(x, w, bias=None): 
    H = w @ x

    I_vect = np.ones(w.shape[0])
    
    return H, I_vect
    


def get_I_relu(x, w, bias=None, alpha=0.3):
    W_x = w @ x
    
    mul_pos = W_x > 0
    mul_pos = mul_pos * 1
    
    mul = mul_pos
    
    if alpha > 0:
        mul_neg = W_x <= 0
        mul_neg = mul_neg * alpha  
        mul = mul_pos + mul_neg 
          
    I_vect = mul.ravel()

    H = W_x * mul
    
    return H, I_vect
    
        

'''
Hard_sigmoid
    np.maximum(0, np.minimum(1, x))
'''
def get_I_hard_sigmoid(x, w, bias=None):
    
    W_x = w @ x
    
    # mul_inf = W_x < 0
    mul_sup = W_x > 1
    mul_sup = np.divide(1, W_x, where=mul_sup, out=np.zeros(W_x.shape))
    
    mul_else = (W_x >= 0) & (W_x <= 1)
    mul_else = mul_else * 1
    
    mul = mul_else + mul_sup
        
    I_vect = mul.ravel()

    H = W_x * mul
    
    return H, I_vect
    
  

'''
Hard_sigmoid
    np.maximum(0, np.minimum(1, x))
'''
def hard_sigmoid(x):
    zeros = tf.zeros_like(x)
    ones = tf.ones_like(x)
    return tf.math.maximum(zeros, tf.math.minimum(ones, x))


'''
Hard_tanh
     np.maximum(-1, np.minimum(1, x))
'''
def get_I_hard_tanh(x, w, bias=None):
    
    W_x = w @ x
    
    mul_plus_one = W_x > 1
    mul_less_one = W_x < -1
    
    mul_else = (W_x >= -1) & (W_x <= 1)
    mul_else = mul_else * 1
    
    mul_sup = np.divide(1, W_x, where=mul_plus_one, out=np.zeros(W_x.shape))
    
    mul_inf = np.divide(-1, W_x, where=mul_less_one, out=np.zeros(W_x.shape))
    
    mul = mul_inf + mul_else + mul_sup
        
    I_vect = mul.ravel()
    
    H = W_x * mul
    
    return H, I_vect


'''
Hard_tanh
    np.maximum(-1, np.minimum(1, x))
'''
def hard_tanh(x):
  ones = tf.ones_like(x)
  return tf.math.maximum(-ones, tf.math.minimum(ones, x))



'''
Obtemos la matriz ampliada con el bias incorporado
'''
def get_transposed_ext(w, bias=None):
    if not isinstance(bias, np.ndarray):
        bias = np.zeros(w.shape[1])
      
    bias_T = bias.reshape(-1, 1)
    
    zeros = np.zeros(w.shape[0] + 1)
    zeros[0] = 1
    
    return np.vstack((zeros, np.hstack((bias_T, w.T))))
  

