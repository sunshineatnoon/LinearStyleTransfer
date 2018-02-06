from __future__ import division
import torch
from torch.utils.serialization import load_lua
import scipy.misc
import numpy as np
import time
from numpy.lib.stride_tricks import as_strided
import os
from PIL import Image
import cv2
# TODO: prerequites of imageio
import imageio

def numpy2cv2(cont,style,prop,width,height):
    cont = cont.transpose((1,2,0))
    cont = cont[...,::-1]
    cont = cont * 255
    cont = cv2.resize(cont,(width,height))
    #cv2.resize(iimg,(width,height))
    style = style.transpose((1,2,0))
    style = style[...,::-1]
    style = style * 255
    style = cv2.resize(style,(width,height))

    prop = prop.transpose((1,2,0))
    prop = prop[...,::-1]
    prop = prop * 255
    prop = cv2.resize(prop,(width,height))

    return np.concatenate((cont,np.concatenate((style,prop),axis=1)),axis=1)

def makeVideo(content,styles,props,name):
    layers,height,width = content[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(name+'.avi',fourcc,10.0,(width*3,height))
    GIFWriter = imageio.get_writer(name+'.gif',mode='I')
    for j in range(len(content)):
        cv2.imwrite('test.png',numpy2cv2(content[j],styles[j],props[j],width,height))
        # TODO: this is ugly, fix this
        imgj = cv2.imread('test.png')

        video.write(imgj)
        # RGB or BRG, yuks
        GIFWriter.append_data(imgj[...,::-1])
    video.release()

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
