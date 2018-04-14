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
parser.add_argument("--vgg_dir", default='models/vgg_normalised_conv4_1.t7', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/feature_invertor_conv4_1.t7', help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r41.pth', help='pre-trained model path')
parser.add_argument("--stylePath", default="data/style/", help='path to style image')
parser.add_argument("--contentPath", default="data/content/", help='path to frames')
parser.add_argument("--outf", default="Artistic/", help='path to transferred images')
parser.add_argument("--batchSize", type=int,default=1, help='batch size')
parser.add_argument('--loadSize', type=int, default=256, help='scale image size')
parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
parser.add_argument("--layer", default="r41", help='which features to transfer, either r31 or r41')

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
                                             num_workers = 1)
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize,test=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                           batch_size = opt.batchSize,
                                           shuffle = False,
                                           num_workers = 1)

################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)

if(opt.layer == 'r31'):
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

for ci,(content,contentName) in enumerate(content_loader):
    contentName = contentName[0]
    contentV.data.resize_(content.size()).copy_(content)
    for sj,(style,styleName) in enumerate(style_loader):
        styleName = styleName[0]
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
