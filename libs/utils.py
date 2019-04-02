from __future__ import division
import os
import cv2
import time
import torch
import scipy.misc
import numpy as np
import scipy.sparse
from PIL import Image
import scipy.sparse.linalg
from cv2.ximgproc import jointBilateralFilter
from torch.utils.serialization import load_lua
from numpy.lib.stride_tricks import as_strided

def whiten(cF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)
    return whiten_cF

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

    #return np.concatenate((cont,np.concatenate((style,prop),axis=1)),axis=1)
    return prop,cont

def makeVideo(content,style,props,outf):
    print('Stack transferred frames back to video...')
    layers,height,width = content[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(os.path.join(outf,'transfer.avi'),fourcc,10.0,(width,height))
    ori_video = cv2.VideoWriter(os.path.join(outf,'content.avi'),fourcc,10.0,(width,height))
    for j in range(len(content)):
        prop,cont = numpy2cv2(content[j],style,props[j],width,height)
        cv2.imwrite('prop.png',prop)
        cv2.imwrite('content.png',cont)
        # TODO: this is ugly, fix this
        imgj = cv2.imread('prop.png')
        imgc = cv2.imread('content.png')

        video.write(imgj)
        ori_video.write(imgc)
        # RGB or BRG, yuks
    video.release()
    ori_video.release()
    os.remove('prop.png')
    os.remove('content.png')
    print('Transferred video saved at %s.'%outf)

def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.outf)
    os.makedirs(expr_dir,exist_ok=True)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
