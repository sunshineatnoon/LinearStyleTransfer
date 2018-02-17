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
parser.add_argument("--vgg_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv3_1.t7', help='maybe print interval')
parser.add_argument("--decoder_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv3_1.t7', help='maybe print interval')
parser.add_argument("--new_decoder_dir", default='AutoEncoder/dec.pth', help='maybe print interval')
parser.add_argument("--compress_dir", default='AutoEncoder/compress.pth', help='maybe print interval')
parser.add_argument("--unzip_dir", default='AutoEncoder/unzip.pth', help='maybe print interval')
parser.add_argument("--contentPath", default="data/content/", help='folder to training image')
parser.add_argument("--outf", default="AETest/", help='folder to output images and model checkpoints')
parser.add_argument("--batchSize", type=int,default=1, help='batch size')
parser.add_argument('--loadSize', type=int, default=256, help='image size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--mode",default="withCU",help="withCU|withoutCU")
parser.add_argument("--layer",default="r31",help="r31|r41")

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

################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)

if(opt.layer == 'r31'):
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)

if(opt.mode == 'withCU'):
    dec.load_state_dict(torch.load(opt.new_decoder_dir))
    compress = nn.Conv2d(256,32,1,1,0)
    unzip = nn.Conv2d(32,256,1,1,0)
    compress.load_state_dict(torch.load(opt.compress_dir))
    unzip.load_state_dict(torch.load(opt.unzip_dir))
################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize),volatile=False)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    contentV = contentV.cuda()
    if(opt.mode == 'withCU'):
        compress.cuda()
        unzip.cuda()

totalTime = 0
imageCounter = 0
for ci,(content,content256,contentName) in enumerate(content_loader):
    contentName = contentName[0]
    contentV.data.resize_(content.size()).copy_(content)
    cF = vgg(contentV)
    if(opt.layer == 'r41'):
        cF = cF[opt.layer]
        
    if(opt.mode == 'withCU'):
        ccF = compress(cF)
        ucF = unzip(ccF)

        transfer = dec(ucF)
        transfer = transfer.clamp(0,1)
        vutils.save_image(transfer.data,'%s/%sCU.png'%(opt.outf,contentName),normalize=True,scale_each=True,nrow=opt.batchSize)
    else:
        transfer = dec(cF)
        transfer = transfer.clamp(0,1)
        vutils.save_image(transfer.data,'%s/%s.png'%(opt.outf,contentName),normalize=True,scale_each=True,nrow=opt.batchSize)
