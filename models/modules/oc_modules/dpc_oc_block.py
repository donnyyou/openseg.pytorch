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

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
    from bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    
elif torch_ver == '0.3':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn_03'))
    from modules import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')  

from base_oc_block import BaseOC_Context_Module



class DPC_OC_Module(nn.Module):
    """
    Reference: 
        Searching for Efficient Multi-Scale Architectures for Dense Image Prediction .
    """
    def __init__(self, features, out_features):
        super(DPC_OC_Module, self).__init__()
        self.context = nn.Sequential(BaseOC_Context_Module(in_channels=features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features//2, 
                                    dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(6, 12), dilation=(6, 12), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(12, 6), dilation=(12, 6), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(12, 24), dilation=(12, 24), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(24, 12), dilation=(24, 12), bias=False),
                                   InPlaceABNSync(out_features))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):
        context = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat([context, feat2, feat3, feat4, feat5], 1)
        bottle = self.bottleneck(out)
        return bottle