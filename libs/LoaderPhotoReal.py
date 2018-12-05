import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

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

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,contentSegPath,styleSegPath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in os.listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.contentSegPath = contentSegPath
        self.styleSegPath = styleSegPath
        self.fineSize = fineSize
        self.transform = transforms.Compose([
        		transforms.Resize(fineSize),
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
            print('no mask provided, fake a whole black one')
            contentSegImg = Image.new('RGB', (contentImg.size))

        try:
            styleSegImgPath = os.path.join(self.styleSegPath,self.image_list[index])
            styleSegImg = default_loader(styleSegImgPath)
        except :
            print('no mask provided, fake a whole black one')
            styleSegImg = Image.new('RGB', (styleImg.size))


        # Turning segmentation images into masks
        styleSegImg = self.transform(styleSegImg)
        contentSegImg = self.transform(contentSegImg)
        content_masks = ExtractMask(contentSegImg)
        style_masks = ExtractMask(styleSegImg)

        # Preprocess Images
        contentImgArbi = self.transform(contentImg)
        styleImgArbi = self.transform(styleImg)

        return contentImgArbi.squeeze(0),styleImgArbi.squeeze(0),content_masks,style_masks,self.image_list[index]

    def __len__(self):
        return len(self.image_list)
