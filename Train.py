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
from libs.MatrixFixCU import MulLayer
from libs.Criterion import LossCriterion
from torch.utils.serialization import load_lua

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv3_1.t7', help='maybe print interval')
parser.add_argument("--vgg4_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv4_1.t7', help='used for loss network')
parser.add_argument("--decoder_dir", default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv3_1.t7', help='maybe print interval')
parser.add_argument("--stylePath", default="/home/xtli/DATA/wikiArt/train/images/", help='path to style image')
parser.add_argument("--contentPath", default="/home/xtli/DATA/MSCOCO/train2014/images/", help='folder to training image')
parser.add_argument("--new_decoder_dir", default='newAE_64/dec.pth', help='maybe print interval')
parser.add_argument("--compress_dir", default='newAE_64/compress.pth', help='maybe print interval')
parser.add_argument("--unzip_dir", default='newAE_64/unzip.pth', help='maybe print interval')
parser.add_argument("--outf", default="trainingImg/", help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="r41", help='layers for content')
parser.add_argument("--style_layers", default="r41,r31,r21,r11", help='layers for style')
parser.add_argument("--batchSize", type=int,default=8, help='batch size')
parser.add_argument("--niter", type=int,default=100000, help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300, help='image size')
parser.add_argument('--fineSize', type=int, default=256, help='image size')
parser.add_argument("--lr", type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=1e-2, help='style loss weight')
parser.add_argument("--reg_weight", type=float, default=10, help='orthogonal loss weight')
parser.add_argument("--log_interval", type=int, default=100, help='maybe print interval')
parser.add_argument("--save_interval", type=int, default=5000, help='maybe print interval')
parser.add_argument("--mode",default="upsample",help="unpool|upsample")
parser.add_argument("--layer", default="r31", help='r11|r21|r31|r41')

################# PREPARATIONS #################
opt = parser.parse_args()
# turn content layers and style layers to a list
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
opt.cuda = torch.cuda.is_available()
print(opt)

if(opt.mode == 'upsample'):
    from libs.models import encoder1,encoder2,encoder3,encoder4
    from libs.models import decoder1,decoder2,decoder3,decoder4
else:
    from libs.modelsUnpool import encoder1,encoder2,encoder3,encoder4
    from libs.modelsUnpool import decoder1,decoder2,decoder3,decoder4

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

# Loss Network
encoder4_torch = load_lua(opt.vgg4_dir)
vgg4 = loss_network(encoder4_torch)
if(opt.layer == 'r11'):
    matrix = MulLayer('r11',opt.compress_dir,opt.unzip_dir)
    vgg = encoder1(encoder_torch)
    dec = decoder1(decoder_torch)
elif(opt.layer == 'r21'):
    matrix = MulLayer('r21',opt.compress_dir,opt.unzip_dir)
    vgg = encoder2(encoder_torch)
    dec = decoder2(decoder_torch)
elif(opt.layer == 'r31'):
    matrix = MulLayer('r31',opt.compress_dir,opt.unzip_dir)
    vgg = encoder3(encoder_torch)
    dec = decoder3(decoder_torch)
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41',opt.compress_dir,opt.unzip_dir)
    vgg = encoder4(encoder_torch)
    dec = decoder4(decoder_torch)
dec.load_state_dict(torch.load(opt.new_decoder_dir))
print(matrix)
for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,opt.content_layers,opt.style_weight,opt.content_weight)
mse_criterion = nn.MSELoss()
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))
styleV = Variable(torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize))
iden_matrix = torch.eye(64)
iden_matrix = iden_matrix.repeat(opt.batchSize,1)
iden_matrixV = Variable(iden_matrix)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg4.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()
    iden_matrixV = iden_matrixV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*5e-5)

for iteration in range(1,opt.niter+1):
    optimizer.zero_grad()

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

    try:
        style,style256,_ = style_loader.next()
    except IOError:
        style,style256,_ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style,style256,_ = style_loader.next()
    except:
        continue

    # RGB to BGR
    contentV.data.resize_(content.size()).copy_(content)
    styleV.data.resize_(style.size()).copy_(style)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)

    if(opt.mode == 'unpool'):
        feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        if(opt.layer == 'r11'):
            transfer = dec(feature)
        elif(opt.layer == 'r21'):
            transfer = dec(feature,cF['pool_idx'],cF['r12'].size())
        elif(opt.layer == 'r31'):
            transfer = dec(feature,cF['pool_idx'],cF['r12'].size(),cF['pool_idx2'],cF['r22'].size())
        else:
            transfer = dec(feature,cF['pool_idx'],cF['r12'].size(),cF['pool_idx2'],cF['r22'].size(),cF['pool_idx3'],cF['r34'].size())
    else:
        if(opt.layer == 'r41'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature,transmatrix = matrix(cF,sF)
        transfer = dec(feature)

    sF_loss = vgg4(styleV)
    cF_loss = vgg4(contentV)
    tF = vgg4(transfer)

    loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)
    reg_loss = mse_criterion(torch.bmm(transmatrix,torch.transpose(transmatrix,1,2)),iden_matrixV)
    reg_loss = reg_loss * opt.reg_weight
    totalLoss = reg_loss*opt.reg_weight + loss

    # backward & optimization
    totalLoss.backward()
    optimizer.step()
    adjust_learning_rate(optimizer,iteration)

    print('Iteration: [%d/%d] Loss: %f contentLoss: %f styleLoss: %f regLoss %f Learng Rate is %7f'%(opt.niter,iteration,loss.data[0],contentLoss,styleLoss,reg_loss.data[0],optimizer.param_groups[0]['lr']))

    if((iteration) % opt.log_interval == 0):
        transfer = transfer.clamp(0,1)
        concat = torch.cat((content,style,transfer.data.cpu()),dim=0)
        vutils.save_image(concat,'%s/%d.png'%(opt.outf,iteration),normalize=True,scale_each=True,nrow=opt.batchSize)

    if(iteration > 0 and (iteration) % opt.save_interval == 0):
        torch.save(matrix.state_dict(), '%s/%s.pth' % (opt.outf,opt.layer))
