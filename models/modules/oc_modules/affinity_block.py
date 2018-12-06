##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/AlexHex7/Non-local_pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import os
import sys
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
import functools

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    
elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none') 


class Affinity_Module(nn.Module):
    '''
    The implementation for the computation of the pair-wise affinity
    '''
    def __init__(self, in_channels):
        super(Affinity_Module, self).__init__()
        self.in_channels = in_channels
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, bias=False),
            InPlaceABNSync(self.in_channels),
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        # x = self.conv_bn_relu(x)
        x_norm = F.normalize(x, p=2, dim=1) 
        key = x_norm.view(batch_size, self.in_channels, -1)
        query = key.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = F.normalize(sim_map, p=1, dim=2)
        return sim_map
