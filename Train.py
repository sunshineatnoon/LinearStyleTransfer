import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
from libs.models import encoder5 as loss_network

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth',
                    help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--stylePath", default="/home/xtli/DATA/wikiArt/train/images/",
                    help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="/home/xtli/DATA/MSCOCO/train2014/images/",
                    help='path to MSCOCO dataset')
parser.add_argument("--outf", default="trainingOutput/",
                    help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="r41",
                    help='layers for content')
parser.add_argument("--style_layers", default="r11,r21,r31,r41",
                    help='layers for style')
parser.add_argument("--batchSize", type=int,default=8,
                    help='batch size')
parser.add_argument("--niter", type=int,default=100000,
                    help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--lr", type=float, default=1e-4,
                    help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0,
                    help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02,
                    help='style loss weight')
parser.add_argument("--log_interval", type=int, default=500,
                    help='log interval')
parser.add_argument("--gpu_id", type=int, default=0,
                    help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=5000,
                    help='checkpoint save interval')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
opt.cuda = torch.cuda.is_available()
if(opt.cuda):
    torch.cuda.set_device(opt.gpu_id)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True
print_options(opt)

################# DATA #################
content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize)
content_loader_ = torch.utils.data.DataLoader(dataset     = content_dataset,
                                              batch_size  = opt.batchSize,
                                              shuffle     = True,
                                              num_workers = 1,
                                              drop_last   = True)
content_loader = iter(content_loader_)
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize)
style_loader_ = torch.utils.data.DataLoader(dataset     = style_dataset,
                                            batch_size  = opt.batchSize,
                                            shuffle     = True,
                                            num_workers = 1,
                                            drop_last   = True)
style_loader = iter(style_loader_)

################# MODEL #################
vgg5 = loss_network()
if(opt.layer == 'r31'):
    matrix = MulLayer('r31')
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
vgg5.load_state_dict(torch.load(opt.loss_network_dir))

for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,
                          opt.content_layers,
                          opt.style_weight,
                          opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
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

    contentV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)

    if(opt.layer == 'r41'):
        feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
    else:
        feature,transmatrix = matrix(cF,sF)
    transfer = dec(feature)

    sF_loss = vgg5(styleV)
    cF_loss = vgg5(contentV)
    tF = vgg5(transfer)
    loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)

    # backward & optimization
    loss.backward()
    optimizer.step()
    print('Iteration: [%d/%d] Loss: %.4f contentLoss: %.4f styleLoss: %.4f Learng Rate is %.6f'%
         (opt.niter,iteration,loss,contentLoss,styleLoss,optimizer.param_groups[0]['lr']))

    adjust_learning_rate(optimizer,iteration)

    if((iteration) % opt.log_interval == 0):
        transfer = transfer.clamp(0,1)
        concat = torch.cat((content,style,transfer.cpu()),dim=0)
        vutils.save_image(concat,'%s/%d.png'%(opt.outf,iteration),normalize=True,scale_each=True,nrow=opt.batchSize)

    if(iteration > 0 and (iteration) % opt.save_interval == 0):
        torch.save(matrix.state_dict(), '%s/%s.pth' % (opt.outf,opt.layer))
