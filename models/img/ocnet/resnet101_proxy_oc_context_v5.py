##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
affine_par = True

from models.modules.utils.resnet_block import conv3x3, Bottleneck
from models.modules.oc_modules.proxy_oc_block import ProxyOC_Context_Module

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync

    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))

        # extra added layers
        self.context4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                dropout=0, sizes=([1]), proxy_cnt=128)
            )
        self.context3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                dropout=0, sizes=([1]), proxy_cnt=128)
            )
        self.context2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                dropout=0, sizes=([1]), proxy_cnt=128)
            )

        self.context_fuse = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            )

        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        context2 = self.context2(x)
        x = self.layer3(x)
        context3 = self.context3(x)
        x_dsn = self.dsn(context3)
        x = self.layer4(x)
        context4 = self.context4(x)
        x = self.context_fuse(torch.cat([context2, context3, context4], 1))
        x = self.cls(x)
        return [x_dsn, x]


def get_resnet101_proxy_oc_context_dsn_v5(configer):
    model = ResNet(Bottleneck,[3, 4, 23, 3], configer.get('data', 'num_classes'))
    return model