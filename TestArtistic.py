from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from torch.utils.serialization import load_lua
from libs.models import encoder1,encoder2,encoder3,encoder4
from libs.models import decoder1,decoder2,decoder3,decoder4

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_normalised_conv4_1.t7', help='maybe print interval')
parser.add_argument("--decoder_dir", default='models/feature_invertor_conv4_1.t7', help='maybe print interval')
parser.add_argument("--matrixPath", default='models/r41.pth', help='maybe print interval')
parser.add_argument("--stylePath", default="data/style/", help='path to style image')
parser.add_argument("--contentPath", default="data/content/", help='folder to training image')
parser.add_argument("--outf", default="Artistic/", help='folder for transferred images')
parser.add_argument("--batchSize", type=int,default=1, help='batch size')
parser.add_argument('--loadSize', type=int, default=256, help='image size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--layer", default="r41", help='r11|r21|r31|r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

cudnn.benchmark = True

################# DATA #################
content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize,test=True)
content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                             batch_size = opt.batchSize,
                                             shuffle = False,
                                             num_workers = 1,
                                             drop_last = True)
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize,test=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                           batch_size = opt.batchSize,
                                           shuffle = False,
                                           num_workers = 1,
                                           drop_last = True)

################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)

if(opt.layer == 'r11'):
    matrix = MulLayer('r11')
    vgg = encoder1(encoder_torch)
    dec = decoder1(decoder_torch)
elif(opt.layer == 'r21'):
    matrix = MulLayer('r21')
    vgg = encoder2(encoder_torch)
    dec = decoder2(decoder_torch)
elif(opt.layer == 'r31'):
    matrix = MulLayer('r31')
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41')
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)
matrix.load_state_dict(torch.load(opt.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize),volatile=True)
styleV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize),volatile=True)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()
imageCounter = 0
for ci,(content,content256,contentName) in enumerate(content_loader):
    contentName = contentName[0]
    contentV.data.resize_(content.size()).copy_(content)
    for sj,(style,style256,styleName) in enumerate(style_loader):
        styleName = styleName[0]
        # RGB to BGR
        styleV.data.resize_(style.size()).copy_(style)

        # forward
        sF = vgg(styleV)
        cF = vgg(contentV)

        if(opt.layer == 'r41'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature,transmatrix = matrix(cF,sF)
        transfer = dec(feature)

        transfer = transfer.clamp(0,1)
        vutils.save_image(transfer.data,'%s/%s_%s.png'%(opt.outf,contentName,styleName),normalize=True,scale_each=True,nrow=opt.batchSize)
        print('Transferred image saved at %s%s_%s.png'%(opt.outf,contentName,styleName))
