from __future__ import print_function
import argparse
import os
import cv2
import torch
import time
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
from libs.utils import bilateral_filter
from libs.LoaderPhotoReal import Dataset
from libs.MatrixTest import MulLayer
from torch.utils.serialization import load_lua
from libs.models import encoder1,encoder2,encoder3,encoder4
from libs.models import decoder1,decoder2,decoder3,decoder4

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_normalised_conv3_1.t7', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/feature_invertor_conv3_1.t7', help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r31.pth', help='pre-trained model path')
parser.add_argument("--stylePath", default="data/photo_real/style/images/", help='path to style image')
parser.add_argument("--styleSegPath", default="data/photo_real/styleSeg/", help='path to style image masks')
parser.add_argument("--contentPath", default="data/photo_real/content/images/", help='path to content image')
parser.add_argument("--contentSegPath", default="data/photo_real/contentSeg/", help='path to content image masks')
parser.add_argument("--outf", default="PhotoReal/", help='path to save output images')
parser.add_argument("--batchSize", type=int,default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='image size')
parser.add_argument("--layer", default="r31", help='features of which layer to transform, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print(opt)

try:
    os.makedirs(os.path.join(opt.outf,'content'))
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

if(opt.layer == 'r31'):
    matrix = MulLayer(layer='r31')
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)
matrix.load_state_dict(torch.load(opt.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in matrix.parameters():
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

for i,(contentImg,styleImg,cmasks,smasks,imname) in enumerate(loader):
    imname = imname[0]
    contentV.data.resize_(contentImg.size()).copy_(contentImg)
    styleV.data.resize_(styleImg.size()).copy_(styleImg)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)


    if(opt.layer == 'r41'):
        feature = matrix(cF[opt.layer],sF[opt.layer])
    else:
        feature = matrix(cF,sF,cmasks,smasks)
    transfer = dec(feature)

    trans = transfer.data.squeeze(0).mul(255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(trans).save('%s/%s'%(opt.outf,imname))
    content = contentImg.squeeze(0).mul(255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(content).save('%s/content/%s'%(opt.outf,imname))

    filtered = bilateral_filter('%s/%s'%(opt.outf,imname),os.path.join(opt.outf,'content/',imname))
    cv2.imwrite('%s/%s_filtered.png'%(opt.outf,imname), filtered)

    print('Transferred image saved at %s%s, bilateral filtered image saved at %s%s_filtered.png'%(opt.outf,imname,opt.outf,imname))
