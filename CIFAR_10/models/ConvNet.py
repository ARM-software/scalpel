from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
from newLayers import *

class LRN(nn.Module):
    def __init__(self, local_size=1, Alpha=1.0, Beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.Alpha = Alpha
        self.Beta = Beta
        return

    def forward(self, x): 
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.Alpha).add(1.0).pow(self.Beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.Alpha).add(1.0).pow(self.Beta)
        x = x.div(div)
        return x


class ConvNet(nn.Module):
    def __init__(self, prune):
        super(ConvNet, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = LRN(local_size=3, Alpha=5e-5, Beta=0.75, ACROSS_CHANNELS=False)
        if self.prune == 'node':
            self.mask_conv1 = MaskLayer(32, conv=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.norm2 = LRN(local_size=3, Alpha=5e-5, Beta=0.75, ACROSS_CHANNELS=False)
        if self.prune == 'node':
            self.mask_conv2 = MaskLayer(32, conv=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu_conv3 = nn.ReLU(inplace=True)
        if self.prune == 'node':
            self.mask_conv3 = MaskLayer(64, conv=True)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.ip1 = nn.Linear(64*4*4, 10)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x) 
        x = self.pool1(x)
        x = self.norm1(x)
        if self.prune == 'node':
            x = self.mask_conv1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        if self.prune == 'node':
            x = self.mask_conv2(x)
        x = self.conv3(x)
        x = self.relu_conv3(x)
        if self.prune == 'node':
            x = self.mask_conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), 64*4*4)
        x = self.ip1(x)
        return x
