from __future__ import division
import torch
import torch.nn as nn
import cv2
from cv2.ximgproc import createGuidedFilter
from cv2.ximgproc import jointBilateralFilter

class FastPropagator(nn.Module):
  def __init__(self, radius=35, eps=0.3):
    super(FastPropagator, self).__init__()
    self.radius = radius
    self.eps = eps

  def process(self, initImg, contentImg, upsample=False):
    y_img = cv2.imread(initImg)
    if(upsample):
        y_img = cv2.resize(y_img,dsize=(512,512),interpolation=cv2.INTER_LINEAR)
    c_img = cv2.imread(contentImg)
    c_img = cv2.resize(c_img,dsize=(y_img.shape[1],y_img.shape[0]))
    r_img = jointBilateralFilter(c_img,y_img,d=-1,sigmaColor=3,sigmaSpace=20)
    return r_img
