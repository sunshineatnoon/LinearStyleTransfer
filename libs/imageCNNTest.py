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

    def forward(self,x,mask):
        feature = self.convs(x)
        b,c,h,w, = feature.size()

        mask = cv2.resize(mask.numpy(),(h,w),interpolation=cv2.INTER_NEAREST)
        mask = torch.FloatTensor(mask[:,:,0])
        mask = mask.long()
        mask = mask.view(-1)
        fgcmask = (mask==1).nonzero().squeeze(1)
        feature = feature.view(c,-1)
        fgcmask = Variable(fgcmask.cuda(0),requires_grad=False)
        # 16x924
        nonzeros = torch.index_select(feature,1,fgcmask)
        pooledFeature = self.adaPool(nonzeros.unsqueeze(0))

        output = self.fc(pooledFeature.view(1,-1))
        return output

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

    def forward(self,cF,sF,content,style,cmask,smask):
        cmask_backup = cmask.clone()
        smask_backup = smask.clone()

        cb,cc,ch,cw = cF.size()
        cmask = cv2.resize(cmask.numpy(),(cw,ch),interpolation=cv2.INTER_NEAREST)
        cmask = torch.FloatTensor(cmask[:,:,0])
        #cmask = torch.from_numpy(scipy.misc.imresize(cmask.numpy(),(h,w))/255.0)
        cmask = cmask.long()
        cmask = cmask.view(-1)
        fgcmask = (cmask==1).nonzero().squeeze(1)
        fgcmask = Variable(fgcmask.cuda(0),requires_grad=False)

        sb,sc,sh,sw = sF.size()
        smask = cv2.resize(smask.numpy(),(sw,sh),interpolation=cv2.INTER_NEAREST)
        smask = torch.FloatTensor(smask[:,:,0])
        #smask = torch.from_numpy(scipy.misc.imresize(smask.numpy(),(h,w))/255.0)
        smask = smask.long()
        smask = smask.view(-1)
        fgsmask = (smask==1).nonzero().squeeze(1)
        fgsmask = Variable(fgsmask.cuda(0),requires_grad=False)

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sc,-1)
        sFF_select = torch.index_select(sFF,1,fgsmask)
        sMean = torch.mean(sFF_select,dim=1,keepdim=True)
        sMean = sMean.view(1,sc,1,1)
        sMean = sMean.expand_as(cF)

        sMatrix = self.snet(style,smask_backup)
        #sMean = sMean.view(cb,512,1,1)
        #sMean = sMean.expand_as(cF)

        cMatrix = self.cnet(content,cmask_backup)

        sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
        cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)

        # symetric regularization
        #sMatrixReg = torch.bmm(sMatrix,torch.transpose(sMatrix,1,2))
        #cMatrixReg = torch.bmm(cMatrix,torch.transpose(cMatrix,1,2))

        transmatrix = torch.bmm(sMatrix,cMatrix) # (C*C)

        '''
        A = Norm(F^T*F)
        G = T * F * A
        cFF_Norm = cFF.div(torch.max(cFF,dim=1,keepdim=True)[0])
        A = torch.bmm(torch.transpose(cFF_Norm,1,2),cFF_Norm) # (b*N*N)
        #A = A - torch.max(A,dim=0)[0]
        #A = A.div(torch.sum(A,dim=2,keepdim=True))
        # softmax
        A_exp = torch.exp(A)
        A = A_exp.div(torch.sum(A_exp,dim=2,keepdim=True))
        print(A.max(),A.min(),A.mean())
        '''



        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(c,-1)
        cMean = torch.mean(compress_content,dim=1,keepdim=True)
        cMean = cMean.expand_as(compress_content)
        compress_content = compress_content - cMean
        compress_content_select = torch.index_select(compress_content,1,fgcmask)
        transfeatureFG = torch.mm(transmatrix.squeeze(0),compress_content_select)
        transfeature = compress_content.clone()
        transfeature.index_copy_(1,fgcmask,transfeatureFG)
        #transfeature = torch.bmm(transfeature,A)
        out = self.unzip(transfeature.view(b,c,h,w))
        cmask = cmask.view(1,1,h,w)
        cmask = cmask.repeat(1,sMean.size(1),1,1)
        cmask = Variable(cmask.cuda(0),requires_grad=False)
        sMean = sMean * cmask.float()
        #return out + sMean, transmatrix
        return out
