from PIL import Image
from random import randint
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import scipy.misc
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from random import randint
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,dataPath,loadSize,fineSize,test=False):
        super(Dataset,self).__init__()
        self.dataPath = dataPath
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        #self.image_list = sorted(self.image_list)
        #if(test):
        self.transform = transforms.Compose([
        		transforms.Scale(fineSize),
        		transforms.RandomCrop(fineSize),
        		transforms.ToTensor()])
        self.transformPool = transforms.Compose([
                transforms.Scale(fineSize),
                transforms.ToTensor()])
        #self.transform256 = transforms.Compose([
        #		transforms.Scale(256),
        #		transforms.CenterCrop(256),
        #		transforms.ToTensor()])
        #self.transformAbi = transforms.ToTensor()
        #else:
        #    self.transform = transforms.Compose([
        #                        transforms.Scale(fineSize),
        #                        #transforms.RandomCrop(fineSize),
        #                        transforms.RandomHorizontalFlip(),
        #    ])
        #    self.transform256 = transforms.Compose([
        #                        transforms.Scale(256),
        #                        transforms.ToTensor()
        #    ])

    def __getitem__(self,index):
        # figure out image path
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        # load image
        Img = default_loader(dataPath)
        #if(random() < 0.5):
        #    ImgA = self.transformPool(Img)
        #    ImgA = self.adapool(ImgA)
        #else:
        # random crop
        ImgA = self.transform(Img)

        imgName = self.image_list[index]
        imgName = imgName.split('.')[0]

        # generate a mask with 64x64 masked out
        c,h,w = ImgA.size()
        mask = np.ones((h,w))
        left = randint(0,128)
        top = randint(0,128)
        right = left + 128
        bottom = top + 128
        mask[left:right,top:bottom] = 0
        #mask32 = scipy.misc.imresize(mask,(32,32))/255.0
        #mask64 = scipy.misc.imresize(mask,(64,64))/255.0
#
        #mask = torch.from_numpy(mask).long()
        #mask32 = torch.from_numpy(mask32).long()
        #mask64 = torch.from_numpy(mask64).long()

        return ImgA,mask,imgName
        #else:
        #    Img256 = self.transform256(Img)
        #    return transforms.ToTensor()(ImgA),transforms.ToTensor()(ImgA),imgName
            #return ImgA,Img256,imgName

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
