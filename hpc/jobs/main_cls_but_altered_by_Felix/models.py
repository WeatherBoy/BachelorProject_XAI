#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:09:49 2022

@author: manli
"""

'''
imports
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_mds
#--------------------------------------------

'''
Res-Net: for classification
'''


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, stride=stride, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, stride=1, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
            )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                          kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
                )
    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        return F.leaky_relu(out, negative_slope=0.2)

class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 
                            kernel_size=7, 
                            stride=2, bias=False)
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
            )
        self.conv3 = nn.Sequential(
            ResBlock(64, 128, stride=2), 
            ResBlock(128, 128, stride=1)
            )
        self.conv4 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1)
            )
        self.conv5 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1)
            )
        self.conv = nn.Sequential(
            self.conv1, 
            self.conv2, 
            self.conv3, 
            self.conv4, 
            self.conv5
            )
        self.fc = nn.Linear(512, out_channels, bias=False)
        self.dp = nn.Dropout(0.8)
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(batch_size, -1)
        out = self.dp(out)
        return self.fc(out)

if __name__ == "__main__":
    # x = torch.rand(2, 3, 256, 256).cuda()
    # model = UNet(3, 5).cuda()
    # x = model(x)
    # print(x.shape)
    # model = MaskRCNN(3, 14).cuda()
    # x = torch.rand(2, 1, 32, 32)
    # model = EndPointNet()
    # model.eval()
    # offset, quality = model(x)
    # print(x.keys())
    # print(m.keys())

    x = torch.rand(2, 1, 256, 256)
    net = ResNet18()
    x = net(x)
    print(x.shape)
    
    