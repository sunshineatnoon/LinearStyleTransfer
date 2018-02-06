import torch.nn as nn
import torch
import torch.nn.functional as F

'''
class Matrix(nn.Module):
    def __init__(self):
        super(Matrix,self).__init__()
        # 32x32
        self.net1 = nn.Sequential(nn.Conv2d(512,512,3,2,1,groups=512),
                                  nn.ReLU(),
                                  nn.Conv2d(512,512,3,2,1,groups=512),
                                  nn.ReLU(),
                                  nn.Conv2d(512,512,8,1,0,groups=512))
        self.net2 = nn.Sequential(nn.Conv2d(512,512,3,2,1,groups=512),
                                  nn.ReLU(),
                                  nn.Conv2d(512,512,3,2,1,groups=512),
                                  nn.ReLU(),
                                  nn.Conv2d(512,512,8,1,0,groups=512))

        #self.compress = nn.Conv2d(512,64,1,1,0)
        #self.unzip = nn.Conv2d(64,512,1,1,0)

    def forward(self,cF,sF):
        # minus mean
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        #cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMean = sMean.expand_as(cF)
        #sF = sF - sMean

        cbranch = self.net1(cF) # 64x16x16
        sbranch = self.net2(sF) # 64x16x16
        tF = (cF - cMean) * cbranch.expand_as(cF) * sbranch.expand_as(sF) + sMean
        #matrix = self.linear(out.view(out.size(0),-1)) #64x64

        return tF
'''

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3,self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(512,256,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256,128,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,64,3,2,1))
        self.fc = nn.Linear(64*4*4,64*64)

    def forward(self,x):
        out = self.convs(x)
        out = out.view(out.size(0),-1)
        return self.fc(out)

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
        # 64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,64,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,32,3,2,1))
        # 8x8
        self.fc = nn.Linear(32*8*8,64*64)

    def forward(self,x):
        out = self.convs(x)
        out = out.view(out.size(0),-1)
        return self.fc(out)

class Matrix(nn.Module):
    def __init__(self,layer):
        super(Matrix,self).__init__()
        if(layer == 3):
            self.snet = CNN3()
            self.cnet = CNN3()
            self.compress = nn.Conv2d(512,64,1,1,0)
            self.unzip = nn.Conv2d(64,512,1,1,0)
        elif(layer == 2):
            self.snet = CNN2()
            self.cnet = CNN2()
            self.compress = nn.Conv2d(256,64,1,1,0)
            self.unzip = nn.Conv2d(64,256,1,1,0)

    def forward(self,cF,sF):
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
        sMean = sMean.expand_as(cF)
        sF = sF - sMean

        sMatrix = self.snet(sF)
        cMatrix = self.cnet(cF)

        sMatrix = sMatrix.view(sMatrix.size(0),64,64)
        cMatrix = cMatrix.view(cMatrix.size(0),64,64)

        # symetric regularization
        #sMatrixReg = torch.bmm(sMatrix,torch.transpose(sMatrix,1,2))
        #cMatrixReg = torch.bmm(cMatrix,torch.transpose(cMatrix,1,2))

        transmatrix = torch.bmm(sMatrix,cMatrix)

        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)
        transfeature = torch.bmm(transmatrix,compress_content)
        out = self.unzip(transfeature.view(b,c,h,w))
        return out + sMean
