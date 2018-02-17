from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.models as model
from libs.Loader import Dataset
from libs.models import encoder4 as loss_network
from libs.Matrix import MulLayer
from libs.Criterion import LossCriterion
from torch.utils.serialization import load_lua
from libs.models import encoder1,encoder2,encoder3,encoder4
from libs.models import decoder1,decoder2,decoder3,decoder4

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv3_1.t7', help='maybe print interval')
parser.add_argument("--decoder_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv3_1.t7', help='maybe print interval')
parser.add_argument("--stylePath", default="/home/xtli/DATA/wikiArt/train/images/", help='path to style image')
parser.add_argument("--contentPath", default="/home/xtli/DATA/MSCOCO/train2014/images/", help='folder to training image')
parser.add_argument("--outf", default="AutoEncoder/", help='folder to output images and model checkpoints')
parser.add_argument("--batchSize", type=int,default=8, help='batch size')
parser.add_argument("--niter", type=int,default=80000, help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300, help='image size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--lr", type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument("--log_interval", type=int, default=500, help='maybe print interval')
parser.add_argument("--save_interval", type=int, default=5000, help='maybe print interval')
parser.add_argument("--layer", default='r31', help='which layer to compress, use corresponding encoder and decoder')
parser.add_argument("--loss_img_weight", default=100, type=float, help='balance between image and feature reconstruction loss')

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
content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize)
content_loader_ = torch.utils.data.DataLoader(dataset=content_dataset,
					      batch_size = opt.batchSize,
				 	      shuffle = True,
					      num_workers = 1,
					      drop_last = True)
content_loader = iter(content_loader_)

################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)

if(opt.layer == 'r31'):
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
    compress = nn.Conv2d(256,64,1,1,0)
    unzip = nn.Conv2d(64,256,1,1,0)
elif(opt.layer == 'r41'):
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)
    compress = nn.Conv2d(512,64,1,1,0)
    unzip = nn.Conv2d(64,512,1,1,0)

for param in vgg.parameters():
    param.requires_grad = False
################# LOSS & OPTIMIZER #################
mse_criterion = nn.MSELoss()
optimizerc = optim.Adam(compress.parameters(), opt.lr)
optimizeru = optim.Adam(unzip.parameters(), opt.lr)
optimizerdec = optim.Adam(dec.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    compress = compress.cuda()
    unzip = unzip.cuda()
    contentV = contentV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*5e-5)

for iteration in range(1,opt.niter+1):
    optimizerc.zero_grad()
    optimizeru.zero_grad()
    optimizerdec.zero_grad()

    # preprocess data
    try:
        content,content256,_ = content_loader.next()
    except IOError:
        content,content256,_ = content_loader.next()
    except StopIteration:
        content_loader = iter(content_loader_)
        content,content256,_ = content_loader.next()
    except:
        continue

    # RGB to BGR
    contentV.data.resize_(content.size()).copy_(content)

    # forward
    cF = vgg(contentV)
    if(opt.layer == 'r41'):
        cF = cF[opt.layer]
    compressed = compress(cF)
    unziped = unzip(compressed)
    transfer = dec(unziped)

    # MSE Loss on reconstruted images
    loss_img = mse_criterion(transfer,contentV.detach())
    loss_img = loss_img * opt.loss_img_weight
    # MSE Loss on content and transferred feature
    loss_feature = mse_criterion(unziped,cF.detach())
    loss = loss_img + loss_feature

    loss.backward()
    optimizerc.step()
    optimizeru.step()
    optimizerdec.step()
    adjust_learning_rate(optimizerc,iteration)
    adjust_learning_rate(optimizeru,iteration)
    adjust_learning_rate(optimizerdec,iteration)

    print('Iteration: [%d/%d] loss_img: %f loss_feature: %f Learng Rate is %7f'%(opt.niter,iteration,loss_img.data[0],loss_feature.data[0],optimizerc.param_groups[0]['lr']))

    if((iteration) % opt.log_interval == 0):
        transfer = transfer.clamp(0,1)
        concat = torch.cat((content,transfer.data.cpu()),dim=0)
        vutils.save_image(concat,'%s/%d.png'%(opt.outf,iteration),normalize=True,scale_each=True,nrow=opt.batchSize)

    if(iteration > 0 and (iteration) % opt.save_interval == 0):
        torch.save(compress.state_dict(), '%s/compress.pth' % (opt.outf))
        torch.save(unzip.state_dict(), '%s/unzip.pth' % (opt.outf))
        torch.save(dec.state_dict(), '%s/dec.pth' % (opt.outf))
