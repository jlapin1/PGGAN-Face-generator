"""
print("\n".join(["%20s%50s"%(m.shape, m.name) for m in D.weights if m.name[-3]=='l']))

ISSUE: If I change a global variable in IPython, it does not register when running
any of my subroutines.
    example: GAMMA=0.0 -Ipython-> 0.1, in train GAMMA still equals 0.0
    solution: must create variables inside train() function

ISSUE: When I pass a layer to another variable, any changes made on that variable
        will also change the former layer
    example: RGBburn = RGBreg
             RGBburn.trainable=False => RGBreg.trainable=False
"""

BATCH_SIZE = 16
LD = 512

import os
import sys
import numpy as np
import tensorflow as tf
import models
import loss_functions as L
import time
from datasets import disk_to_tensor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# https://github.com/tensorflow/tensorflow/issues/24828
#https://github.com/tensorflow/tensorflow/issues/28081
config=ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

###############################################
############ Training functions ###############
###############################################

def train_step(models, inp, Z, func, ALPHA, DOPT, GOPT, gamma=0.0):
    G,D = models
    with tf.GradientTape(persistent=True) as tape1:
        dloss, gloss, Dreal, Dfake, R1 = func(models, inp, Z, ALPHA, gamma)
    if DOPT!=None:
        dgrads = tape1.gradient(dloss, D.trainable_variables)
        DOPT.apply_gradients(zip(dgrads, D.trainable_variables))
    if GOPT!=None:
        ggrads = tape1.gradient(gloss, G.trainable_variables)
        GOPT.apply_gradients(zip(ggrads, G.trainable_variables))
    return dloss, gloss, Dreal, Dfake, R1

def train(
        alpha,              # bool
        rsz,                # int
        epochs=4,           # int
        bs=16,              # int  
        gamma = 0.0,        # float
        gamma_decay = 1,    # float
        pic=1,              # int
        svwts=True          # bool
        ):
    global FILES
    global FIXED_NOISE
    global MODEL_LOSS
    global PATH
    global MODELS
    global OPTS
    G,D = MODELS
    gOpt,dOpt = OPTS
    train_step_graph = tf.function(train_step)
    
    print("Starting training session lasting %d epochs; Resolution %d; Burn=%s"%(epochs, rsz, alpha))
    trtot = len(FILES)
    steps = trtot//bs + 1 if trtot%bs!=0 else trtot//bs
    Alpha = tf.linspace(0., 1., epochs*steps) if alpha else tf.ones((epochs*steps,), dtype=tf.float32)
    
    for m in range(epochs):
        start_epoch = time.time()
        tots = np.zeros((5))
        for n in range(steps):
            start_split = time.time()
            first = n*bs
            last = (n+1)*bs
            amt = last-first
            
            inp = disk_to_tensor(FILES[first:last], PATH, rsz)
            Z = tf.random.normal((inp.shape[0], 1, 1, LD))
            
            dloss, gloss, d_real, d_fake, R1 = train_step_graph(MODELS, inp, Z, MODEL_LOSS, 
                                                                Alpha[m*steps+n], dOpt, gOpt, gamma)
            
            for o,l in enumerate([dloss, gloss, d_real, d_fake, R1]):
                tots[o] += l.numpy()*amt
            
            sys.stdout.write("\rEpoch %d; Batch %d/%d; alpha=%.2f; Losses (DLoss, Gloss, D(X), D(G(Z)), R1): %6.3f, %6.3f, %6.3f, %6.3f, %6.3f; Batch time: %.1f"%(
                             m+1, n+1, steps, Alpha[m*steps+n].numpy(), dloss.numpy(), gloss.numpy(), d_real.numpy(), d_fake.numpy(), R1.numpy(), time.time()-start_split))
        gamma*=gamma_decay
        sys.stdout.write("\rEpoch %d; Losses (DLoss, GLoss, D(X), D(G(Z)), R1): %6.3f, %6.3f, %6.3f, %6.3f, %6.3f; Time elapsed: %.1f%30s\n"%(
                         m+1, tots[0]/trtot, tots[1]/trtot, tots[2]/trtot, tots[3]/trtot, tots[4]/trtot, time.time()-start_epoch,''))
        if (pic>0) & (m%pic==0):
            generate_images(G, z=FIXED_NOISE, alpha=Alpha[m*steps+n], fn='%d.jpg'%(m+1))
        if svwts:
            G.save_weights('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/celeba/PGGAN/weights/%d/G.wts'%(rsz))
            D.save_weights('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/celeba/PGGAN/weights/%d/D.wts'%(rsz))

def generate_images(G, images=None, z=None, alpha=1, fn=None):
    global LD
    if images==None:
        if z==None:
            z = tf.random.normal((9, 1, 1, LD))
        ans = G(z, alpha=alpha, training=False)*0.5+0.5
    else:
        ans = images[:9]*0.5+0.5
    ans = tf.clip_by_value(ans, 0, 1)
    fig = plt.figure(figsize=(15,15))
    for m in range(9):
        ax = plt.subplot(3, 3, m+1)
        ax.imshow(ans[m])
        ax.set_axis_off()
    
    if fn==None:
        plt.show()
    else:
        fig.savefig('C:/Users/glapi/Desktop/'+fn, dpi=500)
        plt.close()

############################################
################## DATASET #################
############################################

PATH = 'C:/Users/glapi/MyDatasets/celebA/img_align_celeba/'
files = os.listdir(PATH)
SZ = BATCH_SIZE*(len(files)//BATCH_SIZE)
files = files[:SZ]
perm = np.random.permutation(SZ)
FILES = np.array(files)[perm]

###############################################
################## MODELS #####################
###############################################

G = models.Generator(last=256)
D = models.Discriminator(last=256)
MODELS = (G,D)

gOpt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99)
dOpt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99)
OPTS = (gOpt, dOpt)

MODEL_LOSS = L.LS_loss 

################################################
################# TRAINING #####################
################################################
# Fixed noise for 9 examples
if os.path.exists('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/celeba/PGGAN/FIXED_NOISE.csv'):
    hold = np.loadtxt('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/celeba/PGGAN/FIXED_NOISE.csv', delimiter=',')
    hold = hold[:, np.newaxis, np.newaxis, :]
    FIXED_NOISE = tf.constant(hold, dtype=tf.float32)
else:
    FIXED_NOISE = tf.random.normal((9, 1, 1, LD), dtype=tf.float32)

RSZ=4
BURN=False
train(BURN, RSZ, epochs=4, bs=16, gamma=0.1, gamma_decay=1.0)
os.rename('C:/Users/glapi/Desktop/4.jpg', 'C:/Users/glapi/Desktop/%d_%s.jpg'%(RSZ, BURN))

BURN=True
RSZ*=2
G.set_state(RSZ, BURN)
D.set_state(RSZ, BURN)
for m in range(int(np.log2(256)-2)):
    
    train(BURN, RSZ, epochs=4, bs=16, gamma=0.1, gamma_decay=1.0)
    os.rename('C:/Users/glapi/Desktop/4.jpg', 'C:/Users/glapi/Desktop/%d_%s.jpg'%(RSZ, BURN))
    
    BURN=False
    G.set_state(RSZ, BURN)
    D.set_state(RSZ, BURN)
    
    train(BURN, RSZ, epochs=4, bs=16, gamma=0.1, gamma_decay=1.0)
    os.rename('C:/Users/glapi/Desktop/4.jpg', 'C:/Users/glapi/Desktop/%d_%s.jpg'%(RSZ, BURN))
    
    BURN=True
    RSZ*=2
    G.set_state(RSZ, BURN)
    D.set_state(RSZ, BURN)