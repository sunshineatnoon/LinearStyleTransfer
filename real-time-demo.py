import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import makeVideo
import torch.backends.cudnn as cudnn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r31.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r31.pth',
                    help='pre-trained decoder path')
parser.add_argument("--style", default="data/style/in2.jpg",
                    help='path to style image')
parser.add_argument("--matrixPath", default="models/r31.pth",
                    help='path to pre-trained model')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--name",default="transferred_video",
                    help="name of generated video")
parser.add_argument("--layer",default="r31",
                    help="features of which layer to transfer")
parser.add_argument("--outf",default="real_time_demo_output",
                    help="output folder")

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print(opt)
os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True

################# DATA #################
def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Scale(opt.fineSize),
                transforms.ToTensor()])
    return transform(img)
style = loadImg(opt.style).unsqueeze(0)

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
dec.load_state_dict(torch.load(opt.dec_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    style = style.cuda()
    content = content.cuda()

totalTime = 0
imageCounter = 0
result_frames = []
contents = []
styles = []
cap = cv2.VideoCapture(0)
cap.set(3,256)
cap.set(4,512)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(opt.outf,opt.name+'.avi'),fourcc,20.0,(512,256))

with torch.no_grad():
    sF = vgg(style)

while(True):
    ret,frame = cap.read()
    frame = cv2.resize(frame,(512,256),interpolation=cv2.INTER_CUBIC)
    frame = frame.transpose((2,0,1))
    frame = frame[::-1,:,:]
    frame = frame/255.0
    frame = torch.from_numpy(frame.copy()).unsqueeze(0)
    content.data.resize_(frame.size()).copy_(frame)
    with torch.no_grad():
        cF = vgg(content)
        if(opt.layer == 'r41'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature,transmatrix = matrix(cF,sF)
        transfer = dec(feature)
    transfer = transfer.clamp(0,1).squeeze(0).data.cpu().numpy()
    transfer = transfer.transpose((1,2,0))
    transfer = transfer[...,::-1]
    out.write(np.uint8(transfer*255))
    cv2.imshow('frame',transfer)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
