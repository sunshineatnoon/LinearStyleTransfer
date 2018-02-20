import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,layer):
        super(CNN,self).__init__()
        if(layer == 'r11'):
            # 256x64x64
            self.convs = nn.Sequential(nn.Conv2d(64,64,3,2,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,64,3,2,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,32,3,2,1))
        elif(layer == 'r31'):
            # 256x64x64
            self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,64,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,32,3,1,1))
        elif(layer == 'r21'):
            # 128x128x128
            self.convs = nn.Sequential(nn.Conv2d(128,64,3,2,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,32,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32,32,3,1,1))
        elif(layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,32,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(32*32,32*32)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)

class MulLayer(nn.Module):
    def __init__(self,layer):
        super(MulLayer,self).__init__()
        self.snet = CNN(layer)
        self.cnet = CNN(layer)

        if(layer == 'r41'):
            self.compress = nn.Conv2d(512,32,1,1,0)
            self.unzip = nn.Conv2d(32,512,1,1,0)
        elif(layer == 'r31'):
            self.compress = nn.Conv2d(256,32,1,1,0)
            self.unzip = nn.Conv2d(32,256,1,1,0)
        elif(layer == 'r21'):
            self.compress = nn.Conv2d(128,32,1,1,0)
            self.unzip = nn.Conv2d(32,128,1,1,0)
        elif(layer == 'r11'):
            self.compress = nn.Conv2d(64,32,1,1,0)
            self.unzip = nn.Conv2d(32,64,1,1,0)

    def forward(self,cF,sF):
        cFBK = cF.clone()
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS

        sMatrix = self.snet(sF)
        cMatrix = self.cnet(cF)


        sMatrix = sMatrix.view(sMatrix.size(0),32,32)
        cMatrix = cMatrix.view(cMatrix.size(0),32,32)

        transmatrix = torch.bmm(sMatrix,cMatrix)

        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)
        transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
        out = self.unzip(transfeature.view(b,c,h,w))
        out = out + sMeanC
        return out, transmatrix
