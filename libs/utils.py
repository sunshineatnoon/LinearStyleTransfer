import os
import cv2
import torch
import imageio
from cv2.ximgproc import jointBilateralFilter

def numpy2cv2(cont,style,prop,width,height):
    cont = cont.transpose((1,2,0))
    cont = cont[...,::-1]
    cont = cont * 255
    cont = cv2.resize(cont,(width,height))
    style = style.transpose((1,2,0))
    style = style[...,::-1]
    style = style * 255
    style = cv2.resize(style,(width,height))

    prop = prop.transpose((1,2,0))
    prop = prop[...,::-1]
    prop = prop * 255
    prop = cv2.resize(prop,(width,height))

    return prop,cont

def makeVideo(content,style,props,name):
    print('Stack transferred frames back to video...')
    outf = os.path.split(name)[0]
    layers,height,width = content[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(name+'.avi',fourcc,10.0,(width,height))
    ori_video = cv2.VideoWriter(os.path.join(outf,'content.avi'),fourcc,10.0,(width,height))
    GIFWriter = imageio.get_writer(name+'.gif',mode='I')
    for j in range(len(content)):
        prop,cont = numpy2cv2(content[j],style,props[j],width,height)
        cv2.imwrite('prop.png',prop)
        cv2.imwrite('content.png',cont)
        imgj = cv2.imread('prop.png')
        imgc = cv2.imread('content.png')

        video.write(imgj)
        ori_video.write(imgc)
        # RGB or BRG, yuks
        GIFWriter.append_data(imgj[...,::-1])
    video.release()
    ori_video.release
    os.remove('prop.png')
    os.remove('content.png')
    print('Transferred video saved at %s.avi.'%name)

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def bilateral_filter(initImg,contentImg,d=-1,sigmaColor=3,sigmaSpace=20):
    y_img = cv2.imread(initImg)
    c_img = cv2.imread(contentImg)
    c_img = cv2.resize(c_img,dsize=(y_img.shape[1],y_img.shape[0]))
    r_img = jointBilateralFilter(c_img,y_img,d,sigmaColor,sigmaSpace)
    return r_img
