from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as model
from libs.LoaderTest import Dataset
from libs.imageCNN import MulLayer
from torch.utils.serialization import load_lua
from libs.photo_smooth_fast import FastPropagator
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv3_1.t7', help='maybe print interval')
parser.add_argument("--decoder_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv3_1.t7', help='maybe print interval')
parser.add_argument("--stylePath", default="wct_data/style/images/", help='path to style image')
parser.add_argument("--styleSegPath", default="wct_data/styleSeg/", help='path to style image')
parser.add_argument("--contentPath", default="wct_data/content/images/", help='folder to training image')
parser.add_argument("--contentSegPath", default="wct_data/contentSeg/", help='folder to training image')
parser.add_argument("--outf", default="test/", help='folder to output images and model checkpoints')
parser.add_argument("--batchSize", type=int,default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--mode",default="upsample",help="unpool|upsample")
parser.add_argument("--layer", default="r31", help='r11|r21|r31|r41')
parser.add_argument("--matrixPath", default="unpool/r31.pth", help='path to pre-trained weight')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print(opt)

if(opt.mode == 'upsample'):
    from libs.models import encoder1,encoder2,encoder3,encoder4
    from libs.models import decoder1,decoder2,decoder3,decoder4
else:
    from libs.modelsUnpool import encoder1,encoder2,encoder3,encoder4
    from libs.modelsUnpool import decoder1,decoder2,decoder3,decoder4


try:
    os.makedirs(opt.outf)
except OSError:
    pass

cudnn.benchmark = True

################# DATA #################
dataset = Dataset(opt.contentPath,opt.stylePath,opt.contentSegPath,opt.styleSegPath,opt.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)

if(opt.layer == 'r11'):
    matrix = MulLayer(matrixSize=32,layer='11')
    vgg = encoder1(encoder_torch)
    dec = decoder1(decoder_torch)
elif(opt.layer == 'r21'):
    matrix = MulLayer(matrixSize=32,layer='21')
    vgg = encoder2(encoder_torch)
    dec = decoder2(decoder_torch)
elif(opt.layer == 'r31'):
    matrix = MulLayer(matrixSize=32,layer='31')
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    matrix = MulLayer(matrixSize=32,layer='41')
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)
matrix.load_state_dict(torch.load(opt.matrixPath))
vgg.cuda()
dec.cuda()
print(vgg)
print(dec)
for param in vgg.parameters():
    param.requires_grad = False
################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))
styleV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))
contentV = Variable(torch.Tensor(opt.batchSize,3,256,256))
filt = FastPropagator()

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()
################# TRAINING #################
totalTime = 0
totalImg = 0
for i,(contentImg,styleImg,cmasks,smasks,imname) in enumerate(loader):
    print('process image %s....'%(imname))
    imname = imname[0]
    start_time = time.time()
    contentV.data.resize_(contentImg.size()).copy_(contentImg)
    styleV.data.resize_(styleImg.size()).copy_(styleImg)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)

    if(opt.mode == 'unpool'):
        feature = matrix(cF[opt.layer],sF[opt.layer],contentV,styleV)
        if(opt.layer == 'r11'):
            transfer = dec(feature)
        elif(opt.layer == 'r21'):
            transfer = dec(feature,cF['pool_idx'],cF['r12'].size())
        elif(opt.layer == 'r31'):
            transfer = dec(feature,cF['pool_idx'],cF['r12'].size(),cF['pool_idx2'],cF['r22'].size())
        else:
            transfer = dec(feature,cF['pool_idx'],cF['r12'].size(),cF['pool_idx2'],cF['r22'].size(),cF['pool_idx3'],cF['r34'].size())
    else:
        if(opt.layer == 'r41'):
            feature = matrix(cF[opt.layer],sF[opt.layer],contentV,styleV,cmasks,smasks)
        else:
            feature = matrix(cF,sF,contentV,styleV,cmasks,smasks)
        transfer = dec(feature)

    vutils.save_image(transfer.data,'%s/%s'%(opt.outf,imname),nrow=opt.batchSize)
    filtered = filt.process('%s/%s'%(opt.outf,imname),'%s/%s'%(opt.contentPath,imname))
    cv2.imwrite('%s/%s_proped.png'%(opt.outf,imname.split('.')[0]), filtered)
    end_time = time.time()
    if(i > 0):
        totalTime += (end_time - start_time)
        totalImg += 1
print('Average time is %f'%(totalTime/totalImg))
