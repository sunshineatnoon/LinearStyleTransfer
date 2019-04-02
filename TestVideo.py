import os
import torch
import argparse
from PIL import Image
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torch.backends.cudnn as cudnn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
import torchvision.transforms as transforms
from libs.utils import makeVideo, print_options

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r31.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r31.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrix_dir", default="models/r31.pth",
                    help='path to pre-trained model')
parser.add_argument("--style", default="data/style/in2.jpg",
                    help='path to style image')
parser.add_argument("--content_dir", default="data/videos/content/mountain_2/",
                    help='path to video frames')
parser.add_argument('--loadSize', type=int, default=512,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='crop image size')
parser.add_argument("--name",default="transferred_video",
                    help="name of generated video")
parser.add_argument("--layer",default="r31",
                    help="features of which layer to transform")
parser.add_argument("--outf",default="videos",
                    help="output folder")

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True

################# DATA #################
def loadImg(imgPath):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Scale(opt.fineSize),
                transforms.ToTensor()])
    return transform(img)
styleV = loadImg(opt.style).unsqueeze(0)

content_dataset = Dataset(opt.content_dir,
                          loadSize = opt.loadSize,
                          fineSize = opt.fineSize,
                          test     = True,
                          video    = True)
content_loader = torch.utils.data.DataLoader(dataset    = content_dataset,
					                         batch_size = 1,
				 	                         shuffle    = False)

################# MODEL #################
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(layer=opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrix_dir))

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(1,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    styleV = styleV.cuda()
    contentV = contentV.cuda()

result_frames = []
contents = []
style = styleV.squeeze(0).cpu().numpy()
sF = vgg(styleV)

for i,(content,contentName) in enumerate(content_loader):
    print('Transfer frame %d...'%i)
    contentName = contentName[0]
    contentV.resize_(content.size()).copy_(content)
    contents.append(content.squeeze(0).float().numpy())
    # forward
    with torch.no_grad():
        cF = vgg(contentV)

        if(opt.layer == 'r41'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature,transmatrix = matrix(cF,sF)
        transfer = dec(feature)

    transfer = transfer.clamp(0,1)
    result_frames.append(transfer.squeeze(0).cpu().numpy())

makeVideo(contents,style,result_frames,opt.outf)
