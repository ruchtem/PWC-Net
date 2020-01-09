import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from pwc_net import PWCNet

def get_flow_matrix(img1, img2, net):
    im_all = [img1, img2]

    #im_all = [imread(img) for img in [im1_fn, im2_fn]]
    #im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    for i in range(len(im_all)):
        im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
        im_all[_i] = im_all[_i][:, :, ::-1]
        im_all[_i] = 1.0 * im_all[_i]/255.0
        
        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        im_all[_i] = torch.from_numpy(im_all[_i])
        im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
        im_all[_i] = im_all[_i].float()

    with torch.no_grad():    
        im_all = Variable(torch.cat(im_all,1).cuda())

    net = net.cuda()
    net.eval()

    flo = net(im_all)
    flo = flo[0] * 20.0
    # TODO: keep it in torch and cuda
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size 
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
    u_ = cv2.resize(flo[:,:,0],(W,H))
    v_ = cv2.resize(flo[:,:,1],(W,H))
    u_ *= W/ float(W_)
    v_ *= H/ float(H_)
    flo = np.dstack((u_,v_))

    return torch.from_numpy(flo)


def normalize_flow_matrix(flow):
    """Normalizes the flow matrix to [-1, 1] to meet torch.nn.functional.grid_sample() requirement"""
    assert flow.dim() == 3 and flow.size()[2] == 2, "Expecting [heigh, width, 2] as dimensions of flow matrix"
    
    h, w, _ = flow.size()

    xx = torch.arange(0, w).view(1,-1).repeat(h,1)
    yy = torch.arange(0, h).view(-1,1).repeat(1,w)
    xx = xx.view(1,h,w).repeat(1,1,1)
    yy = yy.view(1,h,w).repeat(1,1,1)
    grid = torch.cat((xx,yy),0).float().permute(1, 2, 0)

    flow = Variable(grid) + flow

    # scale grid to [-1,1]
    flow[:,:,0] = 2.0*flow[:,:,0].clone() / max(w-1,1)-1.0
    flow[:,:,1] = 2.0*flow[:,:,1].clone() / max(h-1,1)-1.0

    return flow
