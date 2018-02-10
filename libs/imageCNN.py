import torch.nn as nn
import torch
import torch.nn.functional as F

class imageCNN(nn.Module):
    def __init__(self,matrixSize):
        super(imageCNN,self).__init__()
        # 256x256
        '''
        self.convs = nn.Sequential(nn.Conv2d(3,64,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,128,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,64,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,32,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32,16,3,2,1))
        #self.fc = nn.Linear(16*8*8,64*64)
        self.adaPool = nn.AdaptiveMaxPool2d(256)
        self.fc = nn.Linear(16*8*8,matrixSize*matrixSize)
        '''
        self.convs = nn.Sequential(nn.Conv2d(3,64,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,32,3,2,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32,16,3,2,1))
        self.fc = nn.Linear(16*16*16,matrixSize*matrixSize)
        self.adaPool = nn.AdaptiveMaxPool2d((16,256))


    def forward(self,x):
        #out = self.adaPool(x)
        out = self.convs(x)
        out = self.adaPool(out.view(1,16,32*32))
        out = out.view(out.size(0),-1)
        return self.fc(out)

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

    def forward(self,cF,sF,content,style):
        #cb,cc,ch,cw = cF.size()
        #cFF = cF.view(cb,cc,-1)
        #cMean = torch.mean(cFF,dim=2,keepdim=True)
        #cMean = cMean.unsqueeze(3)
        #cMean = cMean.expand_as(cF)
        #cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMean = sMean.expand_as(cF)

        sMatrix = self.snet(style)
        #sMean = sMean.view(cb,512,1,1)
        #sMean = sMean.expand_as(cF)

        cMatrix = self.cnet(content)

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
        compress_content = compress_content.view(b,c,-1)
        cMean = torch.mean(compress_content,dim=2,keepdim=True)
        cMean = cMean.expand_as(compress_content)
        compress_content = compress_content - cMean
        print(compress_content.mean(),compress_content.var())
        transfeature = torch.bmm(transmatrix,compress_content)
        #transfeature = torch.bmm(transfeature,A)
        out = self.unzip(transfeature.view(b,c,h,w))
        return out + sMean, transmatrix
