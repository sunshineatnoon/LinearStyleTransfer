from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from torch.utils.serialization import load_lua
from libs.utils import makeVideo

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv3_1.t7', help='maybe print interval')
parser.add_argument("--decoder_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv3_1.t7', help='maybe print interval')
parser.add_argument("--stylePath", default="data/videos/style/", help='path to style image')
parser.add_argument("--contentPath", default="data/videos/content/ambush_1/", help='folder to training image')
parser.add_argument("--matrixPath", default="weights/r31.pth", help='path to pre-trained model')
parser.add_argument('--loadSize', type=int, default=256, help='image size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--mode",default="upsample",help="upsample|unpool")
parser.add_argument("--name",default="test",help="name of generated video")
parser.add_argument("--layer",default="r31",help="which layer")

################# PREPARATIONS #################
opt = parser.parse_args()
# turn content layers and style layers to a list
opt.cuda = torch.cuda.is_available()
print(opt)

cudnn.benchmark = True
if(opt.mode == 'upsample'):
    from libs.models import encoder1,encoder2,encoder3,encoder4
    from libs.models import decoder1,decoder2,decoder3,decoder4
else:
    from libs.modelsUnpool import encoder1,encoder2,encoder3,encoder4
    from libs.modelsUnpool import decoder1,decoder2,decoder3,decoder4

################# DATA #################
content_dataset = Dataset(opt.contentPath,loadSize=opt.loadSize,fineSize=opt.fineSize,test=True,video=True)
content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
					      batch_size = 1,
				 	      shuffle = False)
style_dataset = Dataset(opt.stylePath,loadSize=opt.loadSize,fineSize=opt.fineSize,test=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
					      batch_size = 1,
				 	      shuffle = False)
################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)

if(opt.layer == 'r11'):
    matrix = MulLayer(layer='r11')
    vgg = encoder1(encoder_torch)
    dec = decoder1(decoder_torch)
elif(opt.layer == 'r21'):
    matrix = MulLayer(layer='r21')
    vgg = encoder2(encoder_torch)
    dec = decoder2(decoder_torch)
elif(opt.layer == 'r31'):
    matrix = MulLayer(layer='r31')
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)
print(matrix)
matrix.load_state_dict(torch.load(opt.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(1,3,opt.fineSize,opt.fineSize),volatile=True)
styleV = Variable(torch.Tensor(1,3,opt.fineSize,opt.fineSize),volatile=True)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    styleV = styleV.cuda()
    contentV = contentV.cuda()

totalTime = 0
imageCounter = 0
result_frames = []
contents = []
styles = []
for i,(content,content256,contentName) in enumerate(content_loader):
    contentName = contentName[0]
    start_time = time.time()
    contentV.data.resize_(content.size()).copy_(content)
    contents.append(content.squeeze(0).float().numpy())
    for j,(style,style256,styleName) in enumerate(style_loader):
        styles.append(style.squeeze(0).float().numpy())
        styleName = styleName[0]
        styleV.data.resize_(style.size()).copy_(style)

        # forward
        sF = vgg(styleV)
        cF = vgg(contentV)


        if(opt.mode == 'unpool'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
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
                feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
            else:
                feature,transmatrix = matrix(cF,sF)
            transfer = dec(feature)

        transfer = transfer.clamp(0,1)
        result_frames.append(transfer.squeeze(0).data.cpu().numpy())
    end_time = time.time()
    if(i > 0):
        totalTime += (end_time - start_time)
        imageCounter += 1

makeVideo(contents,styles,result_frames,opt.name)
print('Processed %d images in %f seconds. Average time is %f seconds'%(imageCounter,totalTime,(totalTime/imageCounter)))
