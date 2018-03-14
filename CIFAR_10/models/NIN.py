from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
from newLayers import *

class NIN(nn.Module):
    def __init__(self, prune=False, beta_initial=0.8002, beta_limit=0.802):
        super(NIN, self).__init__()
        self.prune = prune

        self.conv1      = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.relu_conv1 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_conv1 = MaskLayer(192, True, beta_initial, beta_limit)
        self.cccp1      = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.relu_cccp1 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_cccp1 = MaskLayer(160, True, beta_initial, beta_limit)
        self.cccp2      = nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0)
        self.relu_cccp2 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_cccp2 = MaskLayer(96, True, beta_initial, beta_limit)
        self.pool1      = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout1   = nn.Dropout(0.5)

        self.conv2      = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        self.relu_conv2 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_conv2 = MaskLayer(192, True, beta_initial, beta_limit)
        self.cccp3      = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.relu_cccp3 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_cccp3 = MaskLayer(192, True, beta_initial, beta_limit)
        self.cccp4      = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.relu_cccp4 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_cccp4 = MaskLayer(192, True, beta_initial, beta_limit)
        self.pool2      = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2   = nn.Dropout(0.5)

        self.conv3      = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.relu_conv3 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_conv3 = MaskLayer(192, True, beta_initial, beta_limit)
        self.cccp5      = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.relu_cccp5 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_cccp5 = MaskLayer(192, True, beta_initial, beta_limit)
        self.cccp6      = nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0)
        self.relu_cccp6 = nn.ReLU(inplace=True)
        self.pool3      = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        if self.prune:
            x = self.mask_conv1(x)
        x = self.cccp1(x)
        x = self.relu_cccp1(x)
        if self.prune:
            x = self.mask_cccp1(x)
        x = self.cccp2(x)
        x = self.relu_cccp2(x)
        if self.prune:
            x = self.mask_cccp2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        if self.prune:
            x = self.mask_conv2(x)
        x = self.cccp3(x)
        x = self.relu_cccp3(x)
        if self.prune:
            x = self.mask_cccp3(x)
        x = self.cccp4(x)
        x = self.relu_cccp4(x)
        if self.prune:
            x = self.mask_cccp4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu_conv3(x)
        if self.prune:
            x = self.mask_conv3(x)
        x = self.cccp5(x)
        x = self.relu_cccp5(x)
        if self.prune:
            x = self.mask_cccp5(x)
        x = self.cccp6(x)
        x = self.relu_cccp6(x)
        x = self.pool3(x)
        x = x.view(x.size(0), 10)
        return x
