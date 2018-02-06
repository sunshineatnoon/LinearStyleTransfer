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
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,transferPath,loadSize,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.stylePath = stylePath
        self.transferPath = transferPath
        self.image_list = [x for x in listdir(transferPath) if is_image_file(x)]
        #self.fineSize = fineSize
        #self.loadSize = loadSize
        self.transforms = transforms.Compose([
                            transforms.Scale(loadSize),
                            transforms.RandomCrop(fineSize),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
        ])

    def __getitem__(self,index):
        # figure out image path
        transferPath = os.path.join(self.transferPath,self.image_list[index])
        tmplist = self.image_list[index].split('_')
        contentPath = os.path.join(self.contentPath,self.image_list[index][:27]+'.jpg')
        stylePath = os.path.join(self.stylePath,str(tmplist[3])+'.jpg')

        # load image
        contentImg = default_loader(contentPath)
        styleImg = default_loader(stylePath)
        transferImg = default_loader(transferPath)

        # transform, fix seed to make sure input and target gets same random transformation
        seed = np.random.randint(2147483647)
        random.seed(seed)
        contentImg = self.transforms(contentImg)
        random.seed(seed)
        styleImg = self.transforms(styleImg)
        random.seed(seed)
        transferImg = self.transforms(transferImg)

        return contentImg,styleImg,transferImg

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
