# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:00:11 2020

@author: glapi
"""

import tensorflow as tf
import numpy as np
from scipy import linalg as la

class PFVNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1.e-7):
        super(PFVNorm, self).__init__()
        self.epsilon = epsilon
    
    def call(self, axy):
        divisor = tf.expand_dims(tf.sqrt(tf.reduce_mean(axy**2, axis=-1) + self.epsilon), axis=-1)
        return axy / divisor

class MiniBatch(tf.keras.layers.Layer):
    def __init__(self, div=4):
        super(MiniBatch, self).__init__()
        self.div = div
    
    def call(self, inp):
        batch = tf.shape(inp)[0]
        i = tf.expand_dims(inp, axis=0)
        i = tf.reshape(i, (self.div, -1,)+tuple(inp.shape[1:])) # bs, mini_bs, RSZ, RSZ, feature_maps
        mn = tf.reduce_mean(i, axis=0) # mini_bs, RSZ, RSZ, feature_maps
        mn = tf.reduce_mean((i-mn)**2, axis=0) # mini_bs, RSZ, RSZ, feature_maps
        value = tf.reduce_mean(tf.sqrt(mn+1e-8))
        cfm = value*tf.ones((batch, inp.shape[1], inp.shape[2], 1), dtype=tf.float32)
        return tf.concat([inp, cfm], axis=-1)

def fid_score(imgs1, imgs2):
    I = tf.keras.applications.InceptionV3()
    model = tf.keras.Model(inputs=I.input, outputs=I.get_layer(I.layers[-2].name).output)
    
    act1 = model(imgs1).numpy() # bs, 2048
    act2 = model(imgs2).numpy() # bs, 2048
    
    mean1 = np.mean(act1, axis=0) # 2048,
    mean2 = np.mean(act2, axis=0) # 2048,
    
    act1_ = act1-mean1 # bs, 2048
    act2_ = act2-mean2 # bs, 2048
    
    C1 = act1_.transpose().dot(act1_) # 2048, 2048
    C2 = act2_.transpose().dot(act2_) # 2048, 2048
    
    C12 = C1*C2
    
    d2 = np.sum((mean1-mean2)**2) + np.trace(C1 + C2 - 2*la.sqrtm(C12))
    
    return d2
    