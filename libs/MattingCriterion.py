import numpy as np
import torch
import torch.nn as nn
import scipy.io
import scipy.ndimage as spi
import scipy.sparse as sps
from torch.autograd import Variable
from libs.models import encoder4 as loss_network
from torch.utils.serialization import load_lua
import torchvision.utils as vutils

def getlaplacian1(i_arr, consts, epsilon=1e-5, win_rad=1):
    neb_size = (win_rad * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_rad, w - win_rad):
        for i in range(win_rad, h - win_rad):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(c, 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')[0: l]
    row_inds = row_inds.ravel(order='F')[0: l]
    col_inds = col_inds.ravel(order='F')[0: l]
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse

def getLaplacian(img):
    h, w, _ = img.shape
    coo = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
    indices = np.mat([coo.row, coo.col])

    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(coo.data).float()
    shape = torch.Size(coo.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

class MattingLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, M):
        ctx.save_for_backward(V, M)
        MattingLoss = 0

        V0 = torch.transpose(V[0,:,:],0,1).cpu().contiguous().view(-1)
        V1 = torch.transpose(V[1,:,:],0,1).cpu().contiguous().view(-1)
        V2 = torch.transpose(V[2,:,:],0,1).cpu().contiguous().view(-1)

        MattingLoss_c = torch.mm(V0.unsqueeze(0),torch.mm(M,V0.unsqueeze(-1)))
        MattingLoss += MattingLoss_c
        MattingLoss_c = torch.mm(V1.unsqueeze(0),torch.mm(M,V1.unsqueeze(-1)))
        MattingLoss += MattingLoss_c
        MattingLoss_c = torch.mm(V2.unsqueeze(0),torch.mm(M,V2.unsqueeze(-1)))
        MattingLoss += MattingLoss_c
        return MattingLoss.squeeze(0)

    @staticmethod
    def backward(ctx,grad_output):
        V,M = ctx.saved_variables
        Mdata = M.data
        Vdata = V.data
        c,h,w = Vdata.size()
        V0 = torch.transpose(Vdata[0,:,:],0,1).cpu().contiguous().view(-1)
        V1 = torch.transpose(Vdata[1,:,:],0,1).cpu().contiguous().view(-1)
        V2 = torch.transpose(Vdata[2,:,:],0,1).cpu().contiguous().view(-1)
        grad_V = grad_M = None

        if ctx.needs_input_grad[1]:
            # calculate grad_M
            grad_M = None

        if ctx.needs_input_grad[0]:
            # calculate grad_V
            # 3 x 65536
            grad_V = torch.zeros(V.size())

            grad_V[0,:,:] = 2 * torch.transpose(torch.mm(Mdata,V0.unsqueeze(-1)).view(h,w),0,1)
            grad_V[1,:,:] = 2 * torch.transpose(torch.mm(Mdata,V1.unsqueeze(-1)).view(h,w),0,1)
            grad_V[2,:,:] = 2 * torch.transpose(torch.mm(Mdata,V2.unsqueeze(-1)).view(h,w),0,1)
            grad_V = Variable(grad_V).cuda()
        return grad_V/255,grad_M
