from __future__ import print_function
import os
import argparse

from libs.SPN import SPN
from libs.Loader import Dataset
from libs.models import encoder4
from libs.models import decoder4
from libs.utils import print_options

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--contentPath", default="/home/xtli/DATA/MSCOCO/train2014/images/",
                    help='path to MSCOCO dataset')
parser.add_argument("--outf", default="trainingSPNOutput/",
                    help='folder to output images and model checkpoints')
parser.add_argument("--layer", default="r41",
                    help='layers for content')
parser.add_argument("--batchSize", type=int,default=8,
                    help='batch size')
parser.add_argument("--niter", type=int,default=100000,
                    help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=512,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--lr", type=float, default=1e-3,
                    help='learning rate')
parser.add_argument("--log_interval", type=int, default=500,
                    help='log interval')
parser.add_argument("--save_interval", type=int, default=5000,
                    help='checkpoint save interval')
parser.add_argument("--spn_num", type=int, default=1,
                    help='number of spn filters')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)


os.makedirs(opt.outf, exist_ok = True)

cudnn.benchmark = True

################# DATA #################
content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize)
content_loader_ = torch.utils.data.DataLoader(dataset=content_dataset,
                                              batch_size = opt.batchSize,
                                              shuffle = True,
                                              num_workers = 4,
                                              drop_last = True)
content_loader = iter(content_loader_)

################# MODEL #################
spn = SPN(spn=opt.spn_num)
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))

for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
criterion = nn.MSELoss(size_average=False)
#optimizer_spn = optim.SGD(spn.parameters(), opt.lr)
optimizer_spn = optim.Adam(spn.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    spn.cuda()
    contentV = contentV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)

spn.train()
for iteration in range(1,opt.niter+1):
    optimizer_spn.zero_grad()
    try:
        content,_ = content_loader.next()
    except IOError:
        content,_ = content_loader.next()
    except StopIteration:
        content_loader = iter(content_loader_)
        content,_ = content_loader.next()
    except:
        continue

    contentV.resize_(content.size()).copy_(content)

    # forward
    cF = vgg(contentV)
    transfer = dec(cF['r41'])


    propagated = spn(transfer,contentV)
    loss = criterion(propagated,contentV)

    # backward & optimization
    loss.backward()
    #nn.utils.clip_grad_norm(spn.parameters(), 1000)
    optimizer_spn.step()
    print('Iteration: [%d/%d] Loss: %.4f Learng Rate is %.6f'
         %(opt.niter,iteration,loss,optimizer_spn.param_groups[0]['lr']))

    adjust_learning_rate(optimizer_spn,iteration)

    if((iteration) % opt.log_interval == 0):
        transfer = transfer.clamp(0,1)
        propagated = propagated.clamp(0,1)
        vutils.save_image(transfer,'%s/%d_transfer.png'%(opt.outf,iteration))
        vutils.save_image(propagated,'%s/%d_propagated.png'%(opt.outf,iteration))

    if(iteration > 0 and (iteration) % opt.save_interval == 0):
        torch.save(spn.state_dict(), '%s/%s_spn.pth' % (opt.outf,opt.layer))
