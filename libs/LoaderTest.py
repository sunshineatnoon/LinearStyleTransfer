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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

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

def resize(img,fineSize):
    # TODO: bugs when resize image, need to tell propagation the size
    if((type(fineSize) == int) or fineSize[0] == 0):
        return img
    if(len(fineSize) == 1):
        # we resize the long edge to fineSize and keep image ratio
        w,h = img.size
        if(w > h):
            if(w != fineSize):
                neww = fineSize[0]
                newh = h*neww/w
                img = img.resize((neww,newh))
        else:
            if(h != fineSize):
                newh = fineSize[0]
                neww = w*newh/h
                img = img.resize((neww,newh))
    elif(type(fineSize) == list):
        # we resize both w and h to the given values
        img = img.resize((fineSize[0],fineSize[1]))
    return img

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,contentSegPath,styleSegPath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        # TODO: this could be slow if the video is huge
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.contentSegPath = contentSegPath
        self.styleSegPath = styleSegPath
        self.fineSize = fineSize
        self.transform = transforms.Compose([
        		transforms.Scale((fineSize,fineSize)),
                #transforms.CenterCrop(fineSize),
        		transforms.ToTensor()])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        try:
            contentSegImgPath = os.path.join(self.contentSegPath,self.image_list[index])
            contentSegImg = default_loader(contentSegImgPath)
        except :
            contentSegImg = Image.new('RGB', (contentImg.size))

        try:
            styleSegImgPath = os.path.join(self.styleSegPath,self.image_list[index])
            styleSegImg = default_loader(styleSegImgPath)
        except :
            styleSegImg = Image.new('RGB', (styleImg.size))


        # resize
        #contentImg = resize(contentImg,self.fineSize)
        #styleImg = resize(styleImg,self.fineSize)
        #contentSegImg = resize(contentSegImg,self.fineSize)
        #styleSegImg = resize(styleSegImg,self.fineSize)

        # Turning segmentation images into masks
        styleSegImg = self.transform(styleSegImg)
        contentSegImg = self.transform(contentSegImg)
        content_masks = ExtractMask(contentSegImg)
        style_masks = ExtractMask(styleSegImg)

        # Preprocess Images
        contentImg = self.transform(contentImg)
        styleImg = self.transform(styleImg)

        #cc,ch,cw = contentImg.size()
        #sc,sh,sw = styleImg.size()
        #content_masks = np.ones((ch,cw))
        #style_masks = np.ones((sh,sw))
        return contentImg.squeeze(0),styleImg.squeeze(0),content_masks,style_masks,self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

class VideoDataset(data.Dataset):
    def __init__(self,contentPath,stylePath,contentSegPath,styleSegPath,fineSize):
        super(VideoDataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = sorted([x for x in listdir(contentPath) if is_image_file(x)])
        self.stylePath = stylePath
        self.contentSegPath = contentSegPath
        self.styleSegPath = styleSegPath
        self.fineSize = fineSize
        styleImg = resize(default_loader(stylePath),fineSize)
        if(styleSegPath == None):
            styleSegImg = Image.new('RGB', (styleImg.size))
        else:
            styleSegImg = default_loader(styleSegPath)
        styleSegImg = resize(styleSegImg,fineSize)
        styleSegImg = transforms.ToTensor()(styleSegImg)
        self.styleImg = transforms.ToTensor()(styleImg).unsqueeze(0)
        self.style_masks = ExtractMask(styleSegImg)


    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        contentImg = default_loader(contentImgPath)
        contentImg = resize(contentImg,self.fineSize)

        try:
            contentSegImgPath = os.path.join(self.contentSegPath,self.image_list[index])
            contentSegImg = default_loader(contentSegImgPath)
        except :
            contentSegImg = Image.new('RGB', (contentImg.size))
            # that means we don't have mask, so fake a black image as mask
        contentImg = transforms.ToTensor()(contentImg)

        contentSegImg = resize(contentSegImg,self.fineSize)
        contentSegImg = transforms.ToTensor()(contentSegImg)

        content_masks = ExtractMask(contentSegImg)

        # Preprocess Images
        return contentImg.squeeze(0),content_masks,self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
