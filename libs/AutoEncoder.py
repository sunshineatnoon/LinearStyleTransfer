import torch.nn as nn
import torch
import torch.nn.functional as F

class encoder(torch.nn.Module):
    def __init__(self,vgg,pool='max'):
        super(encoder,self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self,x,style=False,eps=1e-5):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out = F.relu(self.conv4_1(out['p3']))
        return out

class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()

    def forward(self,content,style,eps=1e-6):
        cb,cc,ch,cw = content.size()
        cF = content.view(cb,cc,-1)
        cMean = torch.mean(cF,dim=2,keepdim=True)
        cMean = cMean.expand_as(cF)
        cVar = torch.var(cF,dim=2,keepdim=True)
        cStd = torch.sqrt(cVar + eps)
        cStd = cStd.expand_as(cF)

        sb,sc,sh,sw = style.size()
        sF = style.view(sb,sc,-1)
        sMean = torch.mean(sF,dim=2,keepdim=True)
        sMean = sMean.expand_as(cF)
        sVar = torch.var(sF,dim=2,keepdim=True)
        sStd = torch.sqrt(sVar + eps)
        sStd = sStd.expand_as(cF)

        # normalization
        out = (cF - cMean) / cStd * sStd + sMean
        return out.view(cb,cc,ch,cw)

class decoder(torch.nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x):
        out = F.relu(self.conv1_1(x))
        out = self.upsample1(out)
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = F.relu(self.conv2_3(out))
        out = F.relu(self.conv2_4(out))
        out = self.upsample2(out)
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = self.upsample3(out)
        out = F.relu(self.conv4_1(out))
        out = self.conv4_2(out)
        return out
