import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.misc

class styleLoss(nn.Module):
    def forward(self,input,target):
        iMean = torch.mean(input,dim=1)
        iVar = torch.var(input,dim=1)
        iStd = torch.sqrt(iVar + 1e-6)

        tMean = torch.mean(target,dim=1)
        tVar = torch.var(target,dim=1)
        tStd = torch.sqrt(tVar + 1e-6)

        loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iStd,tStd)
	return loss
'''
class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)

class styleLoss(nn.Module):
    def forward(self,input,target):
        GramTarget = GramMatrix()(target)
        GramInput = GramMatrix()(input)
        return nn.MSELoss()(GramInput,GramTarget)
'''

class LossCriterion(nn.Module):
    def __init__(self,style_layers,content_layers,style_weight,content_weight):
        super(LossCriterion,self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self,tF,sF,cF,smask,cmask):
        #feature = feature.detach()
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.content_layers):
            tf_i = tF[layer]
            b,c,h,w = tf_i.size()

            cmask = torch.from_numpy(scipy.misc.imresize(cmask.numpy(),(h,w))/255.0)
            cmask = cmask.view(-1)
            fgcmask = (cmask==1).nonzero().squeeze(1)
            fgcmask = Variable(fgcmask.cuda(0),requires_grad=False)
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            loss_i = self.contentLosses[i]

            # select out masked features
            tf_i = tf_i.view(c,-1)
            cf_i = cf_i.view(c,-1)
            tf_i_select = torch.index_select(tf_i,1,fgcmask)
            cf_i_select = torch.index_select(cf_i,1,fgcmask)
            totalContentLoss += loss_i(tf_i_select,cf_i_select)
        totalContentLoss = totalContentLoss * self.content_weight

        # style loss
        totalStyleLoss = 0
        for i,layer in enumerate(self.style_layers):
            tf_i = tF[layer]
            b,c,h,w = tf_i.size()

            smask = torch.from_numpy(scipy.misc.imresize(smask.numpy(),(h,w))/255.0)
            smask = smask.view(-1)
            fgsmask = (smask==1).nonzero().squeeze(1)
            fgsmask = Variable(fgsmask.cuda(0),requires_grad=False)
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            loss_i = self.styleLosses[i]

            tf_i = tf_i.view(c,-1)
            sf_i = sf_i.view(c,-1)
            tf_i_select = torch.index_select(tf_i,1,fgsmask)
            sf_i_select = torch.index_select(sf_i,1,fgsmask)
            totalStyleLoss += loss_i(tf_i_select,sf_i_select)
        totalStyleLoss = totalStyleLoss * self.style_weight
        loss = totalStyleLoss + totalContentLoss

        return loss,totalStyleLoss.data[0],totalContentLoss.data[0]
