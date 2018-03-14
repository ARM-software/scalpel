import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../')
from newLayers import *

class AlexNet(nn.Module):

    def __init__(self, prune=None, num_classes=1000,
            beta_initial=0.8002, beta_limit=0.802):
        super(AlexNet, self).__init__()
        self.prune = prune
        self.conv1      = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.bn_conv1   = nn.BatchNorm2d(96, eps=1e-1, momentum=0.1, affine=True)
        self.pool1      = nn.MaxPool2d(kernel_size=3, stride=2)
        if self.prune:
            self.mask_conv1 = MaskLayer(96, conv=True,
                    beta_initial=beta_initial, beta_limit=beta_limit)
        self.conv2      = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.bn_conv2   = nn.BatchNorm2d(256, eps=1e-1, momentum=0.1, affine=True)
        self.pool2      = nn.MaxPool2d(kernel_size=3, stride=2)
        if self.prune:
            self.mask_conv2 = MaskLayer(256, conv=True,
                    beta_initial=beta_initial, beta_limit=beta_limit)
        self.conv3      = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu_conv3 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_conv3 = MaskLayer(384, conv=True,
                    beta_initial=beta_initial, beta_limit=beta_limit)
        self.conv4      = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu_conv4 = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_conv4 = MaskLayer(384, conv=True,
                    beta_initial=beta_initial, beta_limit=beta_limit)
        self.conv5      = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu_conv5 = nn.ReLU(inplace=True)
        self.pool3      = nn.MaxPool2d(kernel_size=3, stride=2)
        if self.prune:
            self.mask_conv5 = MaskLayer(256, conv=True,
                    beta_initial=beta_initial, beta_limit=beta_limit)

        self.ip6        = nn.Linear(256 * 6 * 6, 4096)
        self.relu_ip6   = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_ip6 = MaskLayer(4096,
                    beta_initial=beta_initial, beta_limit=beta_limit)
        self.dropout1   = nn.Dropout()
        self.ip7        = nn.Linear(4096, 4096)
        self.relu_ip7   = nn.ReLU(inplace=True)
        if self.prune:
            self.mask_ip7 = MaskLayer(4096,
                    beta_initial=beta_initial, beta_limit=beta_limit)
        self.dropout2   = nn.Dropout()
        self.ip8        = nn.Linear(4096, num_classes)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.bn_conv1(x)
        x = self.pool1(x)
        if self.prune:
            x = self.mask_conv1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.bn_conv2(x)
        x = self.pool2(x)
        if self.prune:
            x = self.mask_conv2(x)
        x = self.conv3(x)
        x = self.relu_conv3(x)
        if self.prune:
            x = self.mask_conv3(x)
        x = self.conv4(x)
        x = self.relu_conv4(x)
        if self.prune:
            x = self.mask_conv4(x)
        x = self.conv5(x)
        x = self.relu_conv5(x)
        x = self.pool3(x)
        if self.prune:
            x = self.mask_conv5(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.ip6(x)
        x = self.relu_ip6(x)
        if self.prune:
            x = self.mask_ip6(x)
        x = self.dropout1(x)
        x = self.ip7(x)
        x = self.relu_ip7(x)
        if self.prune:
            x = self.mask_ip7(x)
        x = self.dropout2(x)
        x = self.ip8(x)
        return x
