# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

# cost volume: Euclidean Distance
def Cost_Volume(left, right, max_disp):
    N, C, H, W = left.shape
    volume = left.new_zeros([N, C, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            volume[:, :, i, :, i:] = left[:, :, :, i:] - right[:, :, :, :-i]
        else:
            volume[:, :, i, :, :] = left - right
    volume = volume.contiguous()
    return volume

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False), nn.BatchNorm3d(out_channels))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(in_channels, out_channels, 3, stride, pad, dilation),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = convbn(out_channels, out_channels, 3, 1, pad, dilation)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        ori_x = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + ori_x
        out = self.relu2(out)
        return out
