import os
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from libs.SPN import SPN
import torchvision.utils as vutils
from libs.utils import print_options
from libs.MatrixTest import MulLayer
import torch.backends.cudnn as cudnn
from libs.LoaderPhotoReal import Dataset
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
import torchvision.transforms as transforms
from libs.smooth_filter import smooth_filter

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r41.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="data/photo_real/style/images/",
                    help='path to style image')
parser.add_argument("--styleSegPath", default="data/photo_real/styleSeg/",
                    help='path to style image masks')
parser.add_argument("--contentPath", default="data/photo_real/content/images/",
                    help='path to content image')
parser.add_argument("--contentSegPath", default="data/photo_real/contentSeg/",
                    help='path to content image masks')
parser.add_argument("--outf", default="PhotoReal/",
                    help='path to save output images')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='image size')
parser.add_argument("--layer", default="r41",
                    help='features of which layer to transform, either r31 or r41')
parser.add_argument("--spn_dir", default='models/r41_spn.pth',
                    help='path to pretrained SPN model')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf, exist_ok=True)

cudnn.benchmark = True

################# DATA #################
dataset = Dataset(opt.contentPath,opt.stylePath,opt.contentSegPath,opt.styleSegPath,opt.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

################# MODEL #################
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
spn = SPN()
spn.load_state_dict(torch.load(opt.spn_dir))

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
whitenV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    spn.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()
    whitenV = whitenV.cuda()

for i,(contentImg,styleImg,whitenImg,cmasks,smasks,imname) in enumerate(loader):
    imname = imname[0]
    contentV.resize_(contentImg.size()).copy_(contentImg)
    styleV.resize_(styleImg.size()).copy_(styleImg)
    whitenV.resize_(whitenImg.size()).copy_(whitenImg)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)

    with torch.no_grad():
        if(opt.layer == 'r41'):
            feature = matrix(cF[opt.layer],sF[opt.layer],cmasks,smasks)
        else:
            feature = matrix(cF,sF,cmasks,smasks)
        transfer = dec(feature)
        filtered = spn(transfer,whitenV)
    vutils.save_image(transfer,os.path.join(opt.outf,'%s_transfer.png'%(imname.split('.')[0])))

    filtered = filtered.clamp(0,1)
    filtered = filtered.cpu()
    vutils.save_image(filtered,'%s/%s_filtered.png'%(opt.outf,imname.split('.')[0]))
    out_img = filtered.squeeze(0).mul(255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    content = contentImg.squeeze(0).mul(255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    content = content.copy()
    out_img = out_img.copy()
    smoothed = smooth_filter(out_img, content, f_radius=15, f_edge=1e-1)
    smoothed.save('%s/%s_smooth.png'%(opt.outf,imname.split('.')[0]))
    print('Transferred image saved at %s%s, filtered image saved at %s%s_filtered.png' \
          %(opt.outf,imname,opt.outf,imname.split('.')[0]))
