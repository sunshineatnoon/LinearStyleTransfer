from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from libs.utils import whiten

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path,fineSize):
    img = Image.open(path).convert('RGB')
    w,h = img.size
    if(w < h):
        neww = fineSize
        newh = h * neww / w
        newh = int(newh / 8) * 8
    else:
        newh = fineSize
        neww = w * newh / h
        neww = int(neww / 8) * 8
    img = img.resize((neww,newh))
    return img

def MaskHelper(seg,color):
    # green
    mask = torch.Tensor()
    if(color == 'green'):
        mask = torch.lt(seg[0],0.1)
        mask = torch.mul(mask,torch.gt(seg[1],1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2],0.1))
    elif(color == 'black'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'white'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'red'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'blue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'yellow'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'grey'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'lightblue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'purple'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    else:
        print('MaskHelper(): color not recognized, color = ' + color)
    return mask.float()

def ExtractMask(Seg):
    # Given segmentation for content and style, we get a list of segmentation for each color
    '''
    Test Code:
        content_masks,style_masks = ExtractMask(contentSegImg,styleSegImg)
        for i,mask in enumerate(content_masks):
            vutils.save_image(mask,'samples/content_%d.png' % (i),normalize=True)
        for i,mask in enumerate(style_masks):
            vutils.save_image(mask,'samples/style_%d.png' % (i),normalize=True)
    '''
    color_codes = ['blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    masks = []
    for color in color_codes:
        mask = MaskHelper(Seg,color)
        masks.append(mask)
    return masks

def calculate_size(h,w,fineSize):
    if(h > w):
        newh = fineSize
        neww = int(w * 1.0 * newh / h)
    else:
        neww = fineSize
        newh = int(h * 1.0 * neww / w)
    newh = (newh // 8) * 8
    neww = (neww // 8) * 8
    return neww, newh

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,contentSegPath,styleSegPath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.contentSegPath = contentSegPath
        self.styleSegPath = styleSegPath
        self.fineSize = fineSize

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = default_loader(contentImgPath,self.fineSize)
        styleImg = default_loader(styleImgPath,self.fineSize)

        try:
            contentSegImgPath = os.path.join(self.contentSegPath,self.image_list[index])
            contentSegImg = default_loader(contentSegImgPath,self.fineSize)
        except :
            print('no mask provided, fake a whole black one')
            contentSegImg = Image.new('RGB', (contentImg.size))

        try:
            styleSegImgPath = os.path.join(self.styleSegPath,self.image_list[index])
            styleSegImg = default_loader(styleSegImgPath,self.fineSize)
        except :
            print('no mask provided, fake a whole black one')
            styleSegImg = Image.new('RGB', (styleImg.size))


        hs, ws = styleImg.size
        newhs, newws = calculate_size(hs,ws,self.fineSize)

        transform = transforms.Compose([
        		transforms.Resize((newhs, newws)),
        		transforms.ToTensor()])
        # Turning segmentation images into masks
        styleSegImg = transform(styleSegImg)
        styleImgArbi = transform(styleImg)

        hc, wc = contentImg.size
        newhc, newwc = calculate_size(hc,wc,self.fineSize)

        transform = transforms.Compose([
        		transforms.Resize((newhc, newwc)),
        		transforms.ToTensor()])
        contentSegImg = transform(contentSegImg)
        contentImgArbi = transform(contentImg)

        content_masks = ExtractMask(contentSegImg)
        style_masks = ExtractMask(styleSegImg)

        ImgW = whiten(contentImgArbi.view(3,-1).double())
        ImgW = ImgW.view(contentImgArbi.size()).float()

        return contentImgArbi.squeeze(0),styleImgArbi.squeeze(0),ImgW,content_masks,style_masks,self.image_list[index]

    def __len__(self):
        return len(self.image_list)
