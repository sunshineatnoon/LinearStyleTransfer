import torch
import torch.nn as nn

class styleLoss(nn.Module):
    def forward(self,input,target):
        ib,ic,ih,iw = input.size()
        iF = input.view(ib,ic,-1)
        iMean = torch.mean(iF,dim=2)
        iCov = GramMatrix()(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb,tc,-1)
        tMean = torch.mean(tF,dim=2)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iCov,tCov)
        return loss/tb

class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)

class LossCriterion(nn.Module):
    def __init__(self,style_layers,content_layers,style_weight,content_weight):
        super(LossCriterion,self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self,tF,sF,cF):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i,cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # style loss
        totalStyleLoss = 0
        for i,layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i,sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight
        loss = totalStyleLoss + totalContentLoss

        return loss,totalStyleLoss,totalContentLoss
