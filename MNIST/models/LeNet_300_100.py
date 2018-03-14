from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
from newLayers import *


class LeNet_300_100(nn.Module):
    def __init__(self, prune):
        super(LeNet_300_100, self).__init__()
        self.prune = prune
        self.ip1 = nn.Linear(28*28, 300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        if self.prune == 'node':
            self.mask_ip1 = MaskLayer(300)
        self.ip2 = nn.Linear(300, 100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        if self.prune == 'node':
            self.mask_ip2 = MaskLayer(100)
        self.ip3 = nn.Linear(100, 10)
        return

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        if self.prune == 'node':
            x = self.mask_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        if self.prune == 'node':
            x = self.mask_ip2(x)
        x = self.ip3(x)
        return x
