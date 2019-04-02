import torch
from torch.autograd import Function
from .._ext import gaterecurrent2dnoind as gaterecurrent2d

class GateRecurrent2dnoindFunction(Function):
    def __init__(self, horizontal_, reverse_):
        self.horizontal = horizontal_
        self.reverse = reverse_

    def forward(self, X, G1, G2, G3):
        num, channels, height, width = X.size()
        output = torch.zeros(num, channels, height, width)

        if not X.is_cuda:
            print("cpu version is not ready at this time")
            return 0
        else:
            output = output.cuda()
            gaterecurrent2d.gaterecurrent2dnoind_forward_cuda(self.horizontal,self.reverse, X, G1, G2, G3, output)

            self.X = X
            self.G1 = G1
            self.G2 = G2
            self.G3 = G3
            self.output = output
            self.hiddensize = X.size()
            return output

    def backward(self, grad_output):
        assert(self.hiddensize is not None and grad_output.is_cuda)
        num, channels, height, width = self.hiddensize

        grad_X = torch.zeros(num, channels, height, width).cuda()
        grad_G1 = torch.zeros(num, channels, height, width).cuda()
        grad_G2 = torch.zeros(num, channels, height, width).cuda()
        grad_G3 = torch.zeros(num, channels, height, width).cuda()

        gaterecurrent2d.gaterecurrent2dnoind_backward_cuda(self.horizontal, self.reverse, self.output, grad_output, self.X, self.G1, self.G2, self.G3, grad_X, grad_G1, grad_G2, grad_G3)

        del self.hiddensize
        del self.G1
        del self.G2
        del self.G3
        del self.output
        del self.X

        return grad_X, grad_G1, grad_G2, grad_G3
