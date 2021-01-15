# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 07:58:01 2021

@author: glapi
"""

import tensorflow as tf
import tensorflow.keras.layers as L
import pg_functions as PG
import numpy as np

initializer = tf.keras.initializers.GlorotUniform()

def Gblock(filters):
    model = tf.keras.Sequential(name='RegularBlock_%d'%(filters))
    model.add(L.UpSampling2D())
    model.add(tf.keras.layers.Conv2DTranspose(filters, 3, 1, padding='SAME', kernel_initializer=initializer))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(PG.PFVNorm())
    model.add(tf.keras.layers.Conv2DTranspose(filters, 3, 1, padding='SAME', kernel_initializer=initializer))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(PG.PFVNorm())
    return model

class Generator(tf.keras.models.Model):
    def __init__(self,
                 rsz=4,                 # int, Starting resolution
                 burn=False,            # bool, Starting state: burn-in or not
                 last=1024):            # int, Final resolution
        super(Generator, self).__init__()
        self.offset = np.log2(last)-5
        self.block = [
            tf.keras.Sequential([
                L.Conv2DTranspose(512, 4, 1, padding='VALID', kernel_initializer=initializer),
                L.Conv2DTranspose(512, 3, 1, padding='SAME', kernel_initializer=initializer),
                PG.PFVNorm()
                ], name='StartBlock_512')
            ]
        self.RGBreg = L.Conv2D(3,1,1, padding='SAME', kernel_initializer=initializer, name='RGBreg')
        self.set_state(rsz, burn)
        
    def add_block(self, filters):
        self.block.append(Gblock(filters))
    
    def extinguish(self):
        self.burn=False
        self.RGBburn.trainable=False
    
    def set_state(self, rsz, burn=True):
        self.rsz = rsz
        self.burn = burn
        extent = max(0,int(np.log2(rsz)-self.offset))
        if rsz>4:
            if burn:
                self.add_block(min(512, int(512/2**extent)))
                self.RGBburn = self.RGBreg
                # New RGBreg layer
                self.RGBreg = L.Conv2D(3, 1, 1, padding='SAME', kernel_initializer=initializer, name='RGBreg')
            else:
                self.extinguish()
        
    def call(self, inp, alpha=1.0):
        out = inp
        for m in range(len(self.block)-1):
            out = self.block[m](out)
        if self.burn:
            upped = self.RGBburn(L.UpSampling2D()(out))
            return (1-alpha)*upped + alpha*self.RGBreg(self.block[-1](out))
        return self.RGBreg(self.block[-1](out))

def Dblock(filters, ceil=512):
    model = tf.keras.Sequential(name='RegularBlock_%d'%(filters))
    model.add(L.Conv2D(filters, 3, 1, padding='SAME', kernel_initializer=initializer))
    model.add(L.LeakyReLU(0.2))
    model.add(L.Conv2D(min(ceil, 2*filters), 3, 1, padding='SAME', kernel_initializer=initializer))
    model.add(L.LeakyReLU(0.2))
    model.add(L.AveragePooling2D())
    return model

class Discriminator(tf.keras.models.Model):
    def __init__(self, 
                 rsz=4,                 # int, Starting resolution
                 burn=False,            # bool, Starting state: burn-in or not
                 last=1024              # int, Final resolution
                 ):
        # RSZ passed in for the initialization
        super(Discriminator, self).__init__()
        self.offset = np.log2(last)-5
        self.block = [
                tf.keras.Sequential([
                    PG.MiniBatch(),
                    L.Conv2D(512, 3, 1, padding='SAME'),
                    L.LeakyReLU(0.2),
                    L.Conv2D(512, 4, 1, padding='VALID'),
                    L.LeakyReLU(0.2),
                    L.Conv2D(1, 1, 1, padding='SAME')
                    ], name='EndBlock')
        ]
        self.RGBreg = L.Conv2D(512, 1, 1, padding='SAME', kernel_initializer=initializer, name='RGBreg')
        self.set_state(rsz, burn)
    
    def add_block(self, filters):
        self.block.append(Dblock(filters))

    def extinguish(self):
        self.burn=False
        self.RGBburn.trainable=False
    
    def set_state(self, rsz, burn=True):
        self.rsz = rsz
        self.burn = burn
        filters = min(512, int(512/2**(np.log2(rsz)-self.offset)))
        if rsz>4:
            if burn:
                self.RGBburn = self.RGBreg
                self.RGBreg = L.Conv2D(filters, 1, 1, padding='SAME', kernel_initializer=initializer, name='RGBburn')
                self.add_block(filters)
            else:
                self.extinguish()
        
    def call(self, inp, alpha=1.0):
        start = 1
        out = self.RGBreg(inp)
        if self.burn:
            downed = tf.nn.avg_pool2d(self.RGBburn(inp), 2, 2, padding='SAME')
            out = (1-alpha)*downed + alpha*self.block[-1](out)
            start=2
        for m in range(start,len(self.block)+1,1):
            out = self.block[-m](out)
        return out
