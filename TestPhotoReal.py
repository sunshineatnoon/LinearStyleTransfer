import os
import cv2
import torch
import argparse
from PIL import Image
import torch.backends.cudnn as cudnn

from libs.MatrixTest import MulLayer
from libs.utils import bilateral_filter
from libs.LoaderPhotoReal import Dataset
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r31.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r31.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r31.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="data/photo_real/style/images/",
                    help='path to style image')
parser.add_argument("--styleSegPath", default="data/photo_real/styleSeg/",
                    help='path to style image masks')
parser.add_argument("--contentPath", default="data/photo_real/content/images/",
                    help='path to content image')
parser.add_argument("--contentSegPath", default="data/photo_real/contentSeg/",
                    help='path to content image masks')
parser.add_argument("--outf", default="PhotoReal/",
                    help='path to save output images')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='image size')
parser.add_argument("--layer", default="r31",
                    help='features of which layer to transform, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print(opt)
os.makedirs(os.path.join(opt.outf,'content'),exist_ok=True)
cudnn.benchmark = True

################# DATA #################
dataset = Dataset(opt.contentPath,opt.stylePath,opt.contentSegPath,opt.styleSegPath,opt.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

################# MODEL #################
if(opt.layer == 'r31'):
    matrix = MulLayer(layer='r31')
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

for i,(contentImg,styleImg,cmasks,smasks,imname) in enumerate(loader):
    imname = imname[0]
    contentV.data.resize_(contentImg.size()).copy_(contentImg)
    styleV.data.resize_(styleImg.size()).copy_(styleImg)

    # forward
    with torch.no_grad():
        sF = vgg(styleV)
        cF = vgg(contentV)


        if(opt.layer == 'r41'):
            feature = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature = matrix(cF,sF,cmasks,smasks)
        transfer = dec(feature)

    trans = transfer.data.squeeze(0).mul(255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(trans).save('%s/%s'%(opt.outf,imname))
    content = contentImg.squeeze(0).mul(255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(content).save('%s/content/%s'%(opt.outf,imname))

    filtered = bilateral_filter('%s/%s'%(opt.outf,imname),os.path.join(opt.outf,'content/',imname))
    cv2.imwrite('%s/%s_filtered.png'%(opt.outf,imname), filtered)

    print('Transferred image saved at %s%s, bilateral filtered image saved at %s%s_filtered.png'%(opt.outf,imname,opt.outf,imname))
