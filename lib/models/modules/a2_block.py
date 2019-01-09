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
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class _DoubleAttentionBlock(nn.Module):
    '''
    Reference: 
        Chen, Yunpeng, et al. *"A2-Nets: Double Attention Networks."*
    Input:
        N X C X H X W
    Parameters:
        channels       : the dimension of the input feature map
        factor         : channel dimention reduction factor (default = 4)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, channels, factor=4, scale=1, global_cnt=19, bn_type=None):
        super(_DoubleAttentionBlock, self).__init__()
        self.factor = factor
        self.scale = scale
        self.channels = channels
        self.global_cnt = global_cnt
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_gather = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.global_cnt,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.global_cnt, bn_type=bn_type),
            nn.Conv2d(in_channels=self.global_cnt, out_channels=self.global_cnt,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.global_cnt, bn_type=bn_type),
        )
        self.f_distribute = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.global_cnt,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.global_cnt, bn_type=bn_type),
            nn.Conv2d(in_channels=self.global_cnt, out_channels=self.global_cnt,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.global_cnt, bn_type=bn_type),
        )

        self.f_down = nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.factor,
                kernel_size=1, stride=1, padding=0)
        self.f_up = nn.Conv2d(in_channels=self.channels // self.factor, out_channels=self.channels,
                kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.f_up.weight, 0)
        nn.init.constant(self.f_up.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        distribute_map = self.f_distribute(x).view(batch_size, self.global_cnt, -1)
        distribute_map = distribute_map.permute(0, 2, 1) # batch x hw x c
        gather_map = self.f_gather(x).view(batch_size, self.global_cnt, -1) # batch x c x hw
        x_down = self.f_down(x).view(batch_size, self.channels // self.factor, -1)
        x_down = x_down.permute(0, 2, 1) # batch x hw x c
        
        distribute_map = F.softmax(distribute_map, dim=-1) # batch x hw x c, normalize the attention scores over dimension c.
        gather_map = F.softmax(gather_map, dim=-1) # batch x c x hw, normalize the attention scores over dimension hw.

        x_global = torch.matmul(gather_map, x_down) # batch x c x c
        x_update = torch.matmul(distribute_map, x_global) # batch x hw x c
        x_update = x_update.permute(0, 2, 1).contiguous()
        x_update = x_update.view(batch_size, self.channels // self.factor, *x.size()[2:])

        x_up = self.f_up(x_update)
        x_up = F.interpolate(input=x_up, size=(h, w), mode='bilinear', align_corners=True)
        return x_up


class DoubleAttentionBlock2D(_DoubleAttentionBlock):
    def __init__(self, channels, factor=4, scale=1, global_cnt=19, bn_type=None):
        super(DoubleAttentionBlock2D, self).__init__(channels,
                                                    factor,
                                                    scale,
                                                    global_cnt,
                                                    bn_type
                                                    )


class A2_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with double-attention based context information.
    """
    def __init__(self, channels, factor, scale=1, global_cnt=19, dropout=0.1, bn_type=None):
        super(A2_Module, self).__init__()
        self.da_context = DoubleAttentionBlock2D(channels, factor, scale, global_cnt, bn_type)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, feats):
        context = self.da_context(feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output