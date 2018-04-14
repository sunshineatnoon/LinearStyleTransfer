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
from libs.models import encoder1,encoder2,encoder3,encoder4
from libs.models import decoder1,decoder2,decoder3,decoder4
from libs.utils import makeVideo
import torchvision.transforms as transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_normalised_conv3_1.t7', help='maybe print interval')
parser.add_argument("--decoder_dir", default='models/feature_invertor_conv3_1.t7', help='maybe print interval')
parser.add_argument("--style", default="data/style/in2.jpg", help='path to style image')
parser.add_argument("--contentPath", default="data/videos/content/ambush_1/", help='folder to training image')
parser.add_argument("--matrixPath", default="models/r31.pth", help='path to pre-trained model')
parser.add_argument('--loadSize', type=int, default=256, help='image size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--name",default="test",help="name of generated video")
parser.add_argument("--layer",default="r31",help="which layer")
parser.add_argument("--outf",default="videos",help="which layer")

################# PREPARATIONS #################
opt = parser.parse_args()
# turn content layers and style layers to a list
opt.cuda = torch.cuda.is_available()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

cudnn.benchmark = True

################# DATA #################
def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Scale(opt.fineSize),
                transforms.ToTensor()])
    return transform(img)
styleV = Variable(loadImg(opt.style).unsqueeze(0),volatile=True)

content_dataset = Dataset(opt.contentPath,loadSize=opt.loadSize,fineSize=opt.fineSize,test=True,video=True)
content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
					      batch_size = 1,
				 	      shuffle = False)
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
contentV = Variable(torch.Tensor(1,3,opt.fineSize,opt.fineSize),volatile=True)

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
style = styleV.data.squeeze(0).cpu().numpy()
sF = vgg(styleV)

for i,(content,content256,contentName) in enumerate(content_loader):
    print('Transfer frame %d...'%i)
    contentName = contentName[0]
    contentV.data.resize_(content.size()).copy_(content)
    contents.append(content.squeeze(0).float().numpy())
    # forward
    cF = vgg(contentV)

    if(opt.layer == 'r41'):
        feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
    else:
        feature,transmatrix = matrix(cF,sF)
    transfer = dec(feature)

    transfer = transfer.clamp(0,1)
    result_frames.append(transfer.squeeze(0).data.cpu().numpy())

makeVideo(contents,style,result_frames,os.path.join(opt.outf,opt.name))
