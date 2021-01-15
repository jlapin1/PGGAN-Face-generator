# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 08:11:52 2021

@author: glapi
"""

import tensorflow as tf
from PIL import Image
import numpy as np

def normalize(im):
    im = (im-127.5)/127.5
    return im

def random_crop_and_jitter(im, rsz):
    addon = int(0.1171875*rsz)
    im = im.resize((rsz+addon, rsz+addon))
    
    rand = np.random.randint(addon, size=(2,))
    im = im.crop((rand[0], rand[1], rsz+rand[0], rsz+rand[1]))
    
    if np.random.rand()<0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    
    return im

def download(filenames, path):
    out = []
    for m in filenames:
        # Read raw image in
        im = Image.open(path+m)
        # Crop long dimension to short
        (w,h) = im.size
        if w<h:
            crp = (h-w)//2
            im = im.crop((0, crp, w, h-crp))
        else:
            crp = (w-h)//2
            im = im.crop((crp, 0, w-crp, h))
        out.append(im)
    # return a list of PILs
    return out       

def dataset(pils, rsz):
    # list(pils) -> np.array -> tf.constant
    out = np.zeros((len(pils), rsz, rsz, 3), dtype=np.float32)
    for m,n in enumerate(pils):
        # Resize and Random crop and jitter
        n = n.resize((rsz, rsz))
        # n = random_crop_and_jitter(n, rsz)
        out[m] = normalize(np.array(n))
    out = tf.constant(out, tf.float32)
    return out

def disk_to_tensor(Files, Path, Rsz):
    images = download(Files, Path)
    tensor = dataset(images, Rsz)
    return tensor