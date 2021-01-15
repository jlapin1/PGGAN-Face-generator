# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 08:02:36 2021

@author: glapi
"""

import tensorflow as tf

def R1Pen(D, real, alpha):
    with tf.GradientTape() as tape:
        tape.watch(real)
        out = D(real, alpha=alpha, training=True)
    grads = tape.gradient(out, real)
    grads = tf.reshape(grads, (grads.shape[0], -1))
    R1 = tf.reduce_mean(tf.reduce_sum(tf.square(grads), axis=1))
    return out, R1

def GP(D, real, fake):
    epsilon = tf.random.uniform((real.shape[0], 1, 1, 1))
    mixed_image = epsilon*real + (1-epsilon)*fake
    with tf.GradientTape() as tape:
        tape.watch(mixed_image)
        mixed_output = D(mixed_image, training=True)
    grads = tape.gradient(mixed_output, mixed_image)
    grads = tf.reshape(grads, (grads.shape[0], -1))
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1)+1e-8)
    Pen = tf.reduce_mean((norm-1.)**2)
    return Pen

def WGAN_loss(models, real, Z, ALPHA=1, gamma=0):
    G,D = models
    fake = G(Z, alpha=ALPHA, training=True)
    dfake = D(fake, alpha=ALPHA, training=True)
    dreal = D(real, alpha=ALPHA, trainin=True)
    Pen = 0 if gamma==0 else GP(D, real, fake)
    fakemean = tf.reduce_mean(dfake)
    realmean = tf.reduce_mean(dreal)
    gloss = -fakemean
    wassloss = fakemean - realmean
    dloss = wassloss + gamma*Pen
    return wassloss, dloss, gloss, Pen
    
def LS_loss(models, real, Z, ALPHA=1, gamma=0):
    G,D = models
    fake = G(Z, alpha=ALPHA, training=True)
    dfake = D(fake, alpha=ALPHA, training=True)
    if gamma==0:
        dreal = D(real, alpha=ALPHA, training=True)
        R1 = 0
    else:
        dreal, R1 = R1Pen(D, real, ALPHA)
    real_label = tf.ones_like(dreal)#tf.random.normal(dreal.shape, 1)
    fake_label = tf.zeros_like(dfake)#tf.random.normal(dfake.shape, 0)
    gloss = tf.reduce_mean((dfake-real_label)**2)
    dloss_real = tf.reduce_mean((dreal-real_label)**2)
    dloss_fake = tf.reduce_mean((dfake-fake_label)**2)
    dloss = 0.5*(dloss_real + dloss_fake) + gamma*R1
    return dloss, gloss, tf.reduce_mean(dreal), tf.reduce_mean(dfake), R1

def BCE_loss(models, real, Z, ALPHA=1, gamma=0.0):
    G,D = models
    fake = G(Z, alpha=ALPHA, training=True)
    dfake = D(fake, alpha=ALPHA, training=True)
    if gamma==0:
        dreal = D(real, alpha=ALPHA, training=True)
        R1 = 0
    else:
        dreal, R1 = R1Pen(real, ALPHA)
    gloss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(dfake), dfake))
    dloss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(dreal), dreal))
    dloss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(dfake), dfake))
    dloss = 0.5*(dloss_real+dloss_fake) + (gamma/2)*R1
    return dloss, gloss, tf.reduce_mean(dreal), tf.reduce_mean(dfake), R1