import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.misc
import cv2
import torchvision.utils as vutils

class imageCNN(nn.Module):
    def __init__(self,matrixSize):
        super(imageCNN,self).__init__()
        # 256x256
        self.convs = nn.Sequential(nn.Conv2d(3,64,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,32,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32,16,3,2,1))
        self.fc = nn.Linear(16*16*16,matrixSize*matrixSize)
        self.adaPool = nn.AdaptiveMaxPool2d((16,256))

    def forward(self,x,masks):
        feature = self.convs(x)
        b,c,h,w, = feature.size()

        color_code_number = 9
        transMatrices = {}
        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(),(w,h),interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if(torch.sum(mask) >= 10):
                mask = mask.view(-1)
                fgcmask = (mask==1).nonzero().view(-1)
                feature = feature.view(c,-1)
                fgcmask = Variable(fgcmask.cuda(0),requires_grad=False)
                nonzeros = torch.index_select(feature,1,fgcmask)
                pooledFeature = self.adaPool(nonzeros.unsqueeze(0))
                transMatrix = self.fc(pooledFeature.view(1,-1))
                transMatrices[i] = transMatrix.clone()
        return transMatrices


class MulLayer(nn.Module):
    def __init__(self,matrixSize=64,layer='41'):
        super(MulLayer,self).__init__()
        self.snet = imageCNN(matrixSize)
        self.cnet = imageCNN(matrixSize)
        if(layer == '41'):
            self.compress = nn.Conv2d(512,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        elif(layer == '31'):
            self.compress = nn.Conv2d(256,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,256,1,1,0)
        elif(layer == '21'):
            self.compress = nn.Conv2d(128,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,128,1,1,0)
        elif(layer == '11'):
            self.compress = nn.Conv2d(64,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,64,1,1,0)
        self.matrixSize = matrixSize
        self.transmatrix = None
        self.layer = layer

    def forward(self,cF,sF,content,style,cmasks,smasks):
        # calculate transfer matrices for all parts
        compress_content = self.compress(cF)
        cb,cc,ch,cw = compress_content.size()
        sb,sc,sh,sw = sF.size()
        compress_content = compress_content.view(cc,-1)
        cMean = torch.mean(compress_content,dim=1,keepdim=True)
        cMean = cMean.expand_as(compress_content)
        compress_content = compress_content - cMean

        sMatrices = self.snet(style,smasks)
        cMatrices = self.cnet(content,cmasks)

        color_code_number = 9
        finalSMean = Variable(torch.zeros(cF.size()).cuda(0))
        finalSMean = finalSMean.view(256,-1)
        transfeature = compress_content.clone()
        for i in range(color_code_number):
            cmask = cmasks[i].clone().squeeze(0)
            smask = smasks[i].clone().squeeze(0)

            cmask = cv2.resize(cmask.numpy(),(cw,ch),interpolation=cv2.INTER_NEAREST)
            cmask = torch.FloatTensor(cmask)
            cmask = cmask.long()
            smask = cv2.resize(smask.numpy(),(sw,sh),interpolation=cv2.INTER_NEAREST)
            smask = torch.FloatTensor(smask)
            smask = smask.long()
            if(torch.sum(cmask) >= 10 and torch.sum(smask) >= 10
               and (i in sMatrices) and (i in cMatrices)):
                cmask = cmask.view(-1)
                fgcmask = (cmask==1).nonzero().squeeze(1)
                fgcmask = Variable(fgcmask.cuda(0),requires_grad=False)

                smask = smask.view(-1)
                fgsmask = (smask==1).nonzero().squeeze(1)
                fgsmask = Variable(fgsmask.cuda(0),requires_grad=False)

                sFF = sF.view(sc,-1)
                sFF_select = torch.index_select(sFF,1,fgsmask)
                sMean = torch.mean(sFF_select,dim=1,keepdim=True)
                sMean = sMean.view(1,sc,1,1)
                sMean = sMean.expand_as(cF)

                sMatrix = sMatrices[i]
                cMatrix = cMatrices[i]

                sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
                cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)

                transmatrix = torch.bmm(sMatrix,cMatrix) # (C*C)

                compress_content_select = torch.index_select(compress_content,1,fgcmask)
                transfeatureFG = torch.mm(transmatrix.squeeze(0),compress_content_select)
                transfeature.index_copy_(1,fgcmask,transfeatureFG)

                sMean = sMean.contiguous()
                sMean_select = torch.index_select(sMean.view(sc,-1),1,fgcmask)
                finalSMean.index_copy_(1,fgcmask,sMean_select)
                #finalSMean = finalSMean*cmaskV + sMean
        out = self.unzip(transfeature.view(cb,cc,ch,cw))
        return out + finalSMean.view(out.size())
