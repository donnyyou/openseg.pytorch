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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
    from bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    
elif torch_ver == '0.3':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn_03'))
    from modules import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none') 


class HighOrder_Module_v1(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, channels, order=3):
        super(HighOrder_Module_v1, self).__init__()
        self.order = order
        self.key_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )

        self._first_order_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )
        self._second_order_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        key_feats = self.key_transform(feats)
        first_order_feats = self._first_order_transform(key_feats)
        second_order_feats = self._second_order_transform(key_feats**2)
        high_order_feats = feats + first_order_feats + second_order_feats
        high_order_feats = self.relu(high_order_feats)
        return high_order_feats


class HighOrder_Module_v2(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, channels, order=3):
        super(HighOrder_Module_v2, self).__init__()
        self.order = order
        self.key_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )
        self._high_order_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )

    def forward(self, feats):
        key_feats = self.key_transform(feats)
        first_order_feats = key_feats
        second_order_feats = key_feats**2
        high_order_feats = feats + first_order_feats + second_order_feats
        high_order_feats = self._high_order_transform(high_order_feats)
        return high_order_feats



class HighOrder_Module_v3(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, channels, order=3):
        super(HighOrder_Module_v3, self).__init__()
        self.order = order
        self.key_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )
        self._high_order_transform = nn.Sequential(
            nn.Conv2d(channels*3, channels, kernel_size=1, padding=0),
            InPlaceABNSync(channels),
            )

    def forward(self, feats):
        key_feats = self.key_transform(feats)
        first_order_feats = key_feats
        second_order_feats = key_feats**2
        high_order_feats = torch.cat([feats, first_order_feats, second_order_feats], 1)
        high_order_feats = self._high_order_transform(high_order_feats)
        return high_order_feats