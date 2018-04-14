from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from libs.Loader import Dataset
from libs.models import encoder5 as loss_network
from libs.Matrix import MulLayer
from libs.Criterion import LossCriterion
from torch.utils.serialization import load_lua
from libs.models import encoder1,encoder2,encoder3,encoder4
from libs.models import decoder1,decoder2,decoder3,decoder4

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_normalised_conv3_1.t7', help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_normalised_conv5_1.t7', help='used for loss network')
parser.add_argument("--decoder_dir", default='models/feature_invertor_conv3_1.t7', help='pre-trained decoder path')
parser.add_argument("--stylePath", default="/home/xtli/DATA/wikiArt/train/images/", help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="/home/xtli/DATA/MSCOCO/train2014/images/", help='path to MSCOCO dataset')
parser.add_argument("--outf", default="trainingOutput/", help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="r41", help='layers for content')
parser.add_argument("--style_layers", default="r11,r21,r31,r41", help='layers for style')
parser.add_argument("--batchSize", type=int,default=8, help='batch size')
parser.add_argument("--niter", type=int,default=100000, help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300, help='scale image size')
parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02, help='style loss weight')
parser.add_argument("--log_interval", type=int, default=500, help='log interval')
parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')
parser.add_argument("--layer", default="r31", help='which features to transfer, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
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
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize)
style_loader_ = torch.utils.data.DataLoader(dataset=style_dataset,
                                            batch_size = opt.batchSize,
                                            shuffle = True,
                                            num_workers = 1,
                                            drop_last = True)
style_loader = iter(style_loader_)

################# MODEL #################
encoder_torch = load_lua(opt.vgg_dir)
decoder_torch = load_lua(opt.decoder_dir)
encoder4_torch = load_lua(opt.loss_network_dir)

vgg4 = loss_network(encoder4_torch)
if(opt.layer == 'r31'):
    matrix = MulLayer('r31')
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41')
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)

for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,opt.content_layers,opt.style_weight,opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))
styleV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg4.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)

for iteration in range(1,opt.niter+1):
    optimizer.zero_grad()
    try:
        content,_ = content_loader.next()
    except IOError:
        content,_ = content_loader.next()
    except StopIteration:
        content_loader = iter(content_loader_)
        content,_ = content_loader.next()
    except:
        continue

    try:
        style,_ = style_loader.next()
    except IOError:
        style,_ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style,_ = style_loader.next()
    except:
        continue

    contentV.data.resize_(content.size()).copy_(content)
    styleV.data.resize_(style.size()).copy_(style)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)

    if(opt.layer == 'r41'):
        feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
    else:
        feature,transmatrix = matrix(cF,sF)
    transfer = dec(feature)

    sF_loss = vgg4(styleV)
    cF_loss = vgg4(contentV)
    tF = vgg4(transfer)
    loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)

    # backward & optimization
    loss.backward()
    optimizer.step()
    print('Iteration: [%d/%d] Loss: %.4f contentLoss: %.4f styleLoss: %.4f Learng Rate is %.6f'%(opt.niter,iteration,loss.data[0],contentLoss,styleLoss,optimizer.param_groups[0]['lr']))

    adjust_learning_rate(optimizer,iteration)

    if((iteration) % opt.log_interval == 0):
        transfer = transfer.clamp(0,1)
        concat = torch.cat((content,style,transfer.data.cpu()),dim=0)
        vutils.save_image(concat,'%s/%d.png'%(opt.outf,iteration),normalize=True,scale_each=True,nrow=opt.batchSize)

    if(iteration > 0 and (iteration) % opt.save_interval == 0):
        torch.save(matrix.state_dict(), '%s/%s.pth' % (opt.outf,opt.layer))
