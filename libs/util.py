from __future__ import division
import torch
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
import os
from PIL import Image
import torch.nn as nn
import torchvision.utils as vutils
#from modelsNIPS import decoder1,decoder2,decoder3,decoder4,decoder5
#from modelsNIPS import encoder1,encoder2,encoder3,encoder4,encoder5



class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        '''
        # load pre-trained network
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)
        vgg2 = load_lua(args.vgg2)
        decoder2_torch = load_lua(args.decoder2)
        vgg3 = load_lua(args.vgg3)
        decoder3_torch = load_lua(args.decoder3)
        vgg4 = load_lua(args.vgg4)
        decoder4_torch = load_lua(args.decoder4)
        vgg5 = load_lua(args.vgg5)
        decoder5_torch = load_lua(args.decoder5)


        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)
        '''

    def whiten_and_color(self,cF):
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

        '''
        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        '''
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2,cF)

        '''
        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        '''
        return targetFeature

    def transform(self,cF,sF,csF,alpha):
        cF = cF.double()
        sF = sF.double()
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView,sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        csF.data.resize_(ccsF.size()).copy_(ccsF)
        return csF

def save_and_load(Im):
    vutils.save_image(Im,'test.png',padding=0,nrow=1)
    img = Image.open('test.png')
    img = img.resize((256,256))
    img = transforms.ToTensor()(img)
    return img

def MaskHelper(seg,color):
    # green
    mask = torch.Tensor()
    if(color == 'green'):
        mask = torch.lt(seg[0],0.1)
        mask = torch.mul(mask,torch.gt(seg[1],1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2],0.1))
    elif(color == 'black'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'white'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'red'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'blue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'yellow'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'grey'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'lightblue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'purple'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    else:
        print('MaskHelper(): color not recognized, color = ' + color)
    return mask.float()

def ExtractMask(Seg):
    # Given segmentation for content and style, we get a list of segmentation for each color
    '''
    Test Code:
        content_masks,style_masks = ExtractMask(contentSegImg,styleSegImg)
        for i,mask in enumerate(content_masks):
            vutils.save_image(mask,'samples/content_%d.png' % (i),normalize=True)
        for i,mask in enumerate(style_masks):
            vutils.save_image(mask,'samples/style_%d.png' % (i),normalize=True)
    '''
    color_codes = ['blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    masks = []
    for color in color_codes:
        mask = MaskHelper(Seg,color)
        masks.append(mask)
    return masks

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
