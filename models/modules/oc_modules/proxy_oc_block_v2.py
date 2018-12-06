##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/AlexHex7/Non-local_pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
## The Proxy-OC contains a set of features that are used to be the proxy 
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


class _ProxyAttentionBlock(nn.Module):
    '''
    The basic implementation for proxy-attention block
        query branch: the original feature map
        key branch, value branch: shared and both of them are the aggregated features of 
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, proxy_cnt=128):
        super(_ProxyAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.proxy_feat = nn.Parameter(torch.randn(1, key_channels, proxy_cnt))
        self.relu = nn.ReLU(inplace=True)

        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(self.key_channels),
        )

        # key, query, value transform for the proxy features
        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(self.key_channels),
        )
        self.f_value = nn.Sequential(
            nn.Conv1d(in_channels=self.value_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(self.key_channels),
        )
        self.f_key = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(self.key_channels),
        ) 

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        # reduce dimension for x 
        x = self.reduce_dim(x) 
        # normalize the x and proxy
        x = F.normalize(x, p=2, dim=1)
        proxy = F.normalize(self.proxy_feat, p=2, dim=1) 
        # Step 1) x is the key/value and the proxy is the query.
        query = proxy.permute(0, 2, 1)
        key = x.view(batch_size, self.key_channels, -1)
        value = key.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.relu(sim_map)
        sim_map = F.normalize(sim_map, p=1, dim=-1)
        proxy_context = torch.matmul(sim_map, value)
        proxy_context = proxy_context.permute(0, 2, 1).contiguous()
        proxy_context = proxy_context.view(batch_size, *proxy.size()[1:])
        proxy_context = self.f_value(proxy_context)
        proxy_context = F.normalize(proxy_context, p=2, dim=1)

        # Step 2) proxy is the key, proxy_context is the value, x is the query.
        # query = x.view(batch_size, self.key_channels, -1).permute(0, 2, 1) / mIou=76.25
        query = self.f_query(key).permute(0, 2, 1)
        query = F.normalize(query, p=2, dim=2)
        key = proxy
        # key = proxy_context.permute(0, 2, 1) / mIou=76.57
        value = proxy_context.permute(0, 2, 1)  
        sim_map = torch.matmul(query, key)
        sim_map = self.relu(sim_map)
        sim_map = F.normalize(sim_map, p=1, dim=-1)
        x_context = torch.matmul(sim_map, value)
        x_context = x_context.permute(0, 2, 1).contiguous()
        x_context = x_context.view(batch_size, *x.size()[1:])

        context = self.W(x_context)
        if self.scale > 1:
            if torch_ver == '0.4':
                context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
            elif torch_ver == '0.3':
                context = F.upsample(input=context, size=(h, w), mode='bilinear')
        return context


class ProxyAttentionBlock2D(_ProxyAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, proxy_cnt=128):
        super(ProxyAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    value_channels,
                                                    out_channels,
                                                    scale,
                                                    proxy_cnt)


class ProxyOC_Module(nn.Module):
    """
    Implementation of the ProxyOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), proxy_cnt=128):
        super(ProxyOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size, proxy_cnt) for size in sizes])        
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(dropout)
            )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size, proxy_cnt):
        return ProxyAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels, 
                                    size,
                                    proxy_cnt)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ProxyOC_Context_Module(nn.Module):
    """
    Implementation of the ProxyOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), proxy_cnt=128):
        super(ProxyOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size, proxy_cnt) for size in sizes])        
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(dropout)
            )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size, proxy_cnt):
        return ProxyAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels, 
                                    size,
                                    proxy_cnt)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output
