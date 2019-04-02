import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
import sys
sys.path.append('../')
from libs.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

class spn_block(nn.Module):
    def __init__(self, horizontal, reverse):
        super(spn_block, self).__init__()
        self.propagator = GateRecurrent2dnoind(horizontal,reverse)

    def forward(self,x,G1,G2,G3):
        sum_abs = G1.abs() + G2.abs() + G3.abs()
        sum_abs.data[sum_abs.data == 0] = 1e-6
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

        return self.propagator(x,G1,G2,G3)

class VGG(nn.Module):
    def __init__(self,nf):
        super(VGG,self).__init__()
        self.conv1 = nn.Conv2d(3,nf,3,padding = 1)
        # 256 x 256
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv2 = nn.Conv2d(nf,nf*2,3,padding = 1)
        # 128 x 128
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv3 = nn.Conv2d(nf*2,nf*4,3,padding = 1)
        # 64 x 64
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        # 32 x 32
        self.conv4 = nn.Conv2d(nf*4,nf*8,3,padding = 1)

    def forward(self,x):
        output = {}
        output['conv1'] = self.conv1(x)
        x = F.relu(output['conv1'])
        x = self.pool1(x)
        output['conv2'] = self.conv2(x)
        # 128 x 128
        x = F.relu(output['conv2'])
        x = self.pool2(x)
        output['conv3'] = self.conv3(x)
        # 64 x 64
        x = F.relu(output['conv3'])
        output['pool3'] = self.pool3(x)
        # 32 x 32
        output['conv4'] = self.conv4(output['pool3'])
        return output

class Decoder(nn.Module):
    def __init__(self,nf=32,spn=1):
        super(Decoder,self).__init__()
        # 32 x 32
        self.layer0 = nn.Conv2d(nf*8,nf*4,1,1,0) # edge_conv5
        self.layer1 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer2 = nn.Sequential(nn.Conv2d(nf*4,nf*4,3,1,1), # edge_conv8
                                    nn.ELU(inplace=True))
        # 64 x 64
        self.layer3 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer4 = nn.Sequential(nn.Conv2d(nf*4,nf*2,3,1,1), # edge_conv8
                                    nn.ELU(inplace=True))
        # 128 x 128
        self.layer5 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer6 = nn.Sequential(nn.Conv2d(nf*2,nf,3,1,1), # edge_conv8
                                    nn.ELU(inplace=True))
        if(spn == 1):
            self.layer7 = nn.Conv2d(nf,nf*12,3,1,1)
        else:
            self.layer7 = nn.Conv2d(nf,nf*24,3,1,1)
        self.spn = spn
        # 256 x 256

    def forward(self,encode_feature):
        output = {}
        output['0'] = self.layer0(encode_feature['conv4'])
        output['1'] = self.layer1(output['0'])

        output['2'] = self.layer2(output['1'])
        output['2res'] = output['2'] + encode_feature['conv3']
        # 64 x 64

        output['3'] = self.layer3(output['2res'])
        output['4'] = self.layer4(output['3'])
        output['4res'] = output['4'] + encode_feature['conv2']
        # 128 x 128

        output['5'] = self.layer5(output['4res'])
        output['6'] = self.layer6(output['5'])
        output['6res'] = output['6'] + encode_feature['conv1']

        output['7'] = self.layer7(output['6res'])

        return output['7']


class SPN(nn.Module):
    def __init__(self,nf=32,spn=1):
        super(SPN,self).__init__()
        # conv for mask
        self.mask_conv = nn.Conv2d(3,nf,3,1,1)

        # guidance network
        self.encoder = VGG(nf)
        self.decoder = Decoder(nf,spn)

        # spn blocks
        self.left_right = spn_block(True,False)
        self.right_left = spn_block(True,True)
        self.top_down = spn_block(False, False)
        self.down_top = spn_block(False,True)

        # post upsample
        self.post = nn.Conv2d(nf,3,3,1,1)
        self.nf = nf

    def forward(self,x,rgb):
        # feature for mask
        X = self.mask_conv(x)

        # guidance
        features = self.encoder(rgb)
        guide = self.decoder(features)

        G = torch.split(guide,self.nf,1)
        out1 = self.left_right(X,G[0],G[1],G[2])
        out2 = self.right_left(X,G[3],G[4],G[5])
        out3 = self.top_down(X,G[6],G[7],G[8])
        out4 = self.down_top(X,G[9],G[10],G[11])

        out = torch.max(out1,out2)
        out = torch.max(out,out3)
        out = torch.max(out,out4)

        return self.post(out)

if __name__ == '__main__':
    spn = SPN()
    spn = spn.cuda()
    for i in range(100):
        x = Variable(torch.Tensor(1,3,256,256)).cuda()
        rgb = Variable(torch.Tensor(1,3,256,256)).cuda()
        output = spn(x,rgb)
        print(output.size())
