from PIL import Image
from random import randint
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
import random
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,dataPath,loadSize,fineSize,test=False,video=False):
        super(Dataset,self).__init__()
        self.dataPath = dataPath
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.image_list = sorted(self.image_list)
        if(video):
            self.image_list = sorted(self.image_list)
        if not test:
            self.transform = transforms.Compose([
            		     transforms.Scale(fineSize),
            		     transforms.RandomCrop(fineSize),
                         transforms.RandomHorizontalFlip(),
            		     transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            		     transforms.Scale(fineSize),
            		     transforms.ToTensor()])

        self.test = test

    def __getitem__(self,index):
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        Img = default_loader(dataPath)
        ImgA = self.transform(Img)

        imgName = self.image_list[index]
        imgName = imgName.split('.')[0]
        return ImgA,imgName

    def __len__(self):
        return len(self.image_list)
