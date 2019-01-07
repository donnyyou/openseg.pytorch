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

import pdb
import torch
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class _ProxyAttentionBlock(nn.Module):
    '''
    The basic implementation for proxy-attention block,
    we compute (i) a pixel-to-proxy similarity map and 
               (ii)a proxy-to-pixel similarity map.
    For (i), we apply the f_proxy over the proxy features and
    the f_key over the input features.
    For (ii), we apply f_query over the key features.
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, proxy_cnt=128, bn_type=None):
        super(_ProxyAttentionBlock, self).__init__()
        self.proxy_cnt = proxy_cnt
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.proxy_feat = nn.Parameter(torch.randn(1, in_channels, proxy_cnt))
        self.relu = nn.ReLU(inplace=True)

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            # nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
            #     kernel_size=1, stride=1, padding=0),
            # ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )

        # key, query, value transform for the proxy features
        self.f_proxy = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv1d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )

        self.W = nn.Conv2d(in_channels=self.key_channels, out_channels=self.out_channels,
                kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        key = self.f_key(x)
        key = F.normalize(key, p=2, dim=1)
        proxy = self.f_proxy(self.proxy_feat)
        proxy = F.normalize(proxy, p=2, dim=1) 
        query = proxy.permute(0, 2, 1)
        key = key.view(batch_size, self.key_channels, -1)
        value = key.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.relu(sim_map)
        pix2pro_sim_map = F.normalize(sim_map, p=1, dim=-1)

        proxy_context = torch.matmul(pix2pro_sim_map, value)
        proxy_context = proxy_context.permute(0, 2, 1).contiguous()
        proxy_context = proxy_context.view(batch_size, self.key_channels, self.proxy_cnt)

        query = self.f_query(key).permute(0, 2, 1)
        query = F.normalize(query, p=2, dim=2)
        key = proxy
        value = proxy_context.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.relu(sim_map)
        pro2pixel_sim_map = F.normalize(sim_map, p=1, dim=-1)
        x_context = torch.matmul(pro2pixel_sim_map, value)
        x_context = x_context.permute(0, 2, 1).contiguous()
        x_context = x_context.view(batch_size, self.key_channels, *x.size()[2:])

        context = self.W(x_context)
        context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class ProxyAttentionBlock2D(_ProxyAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, proxy_cnt=128, bn_type=None):
        super(ProxyAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    value_channels,
                                                    out_channels,
                                                    scale,
                                                    proxy_cnt, bn_type)


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
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), proxy_cnt=128, bn_type=None):
        super(ProxyOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, 
                                                      key_channels, value_channels, 
                                                      size, proxy_cnt, bn_type) for size in sizes])        
        self.conv_bn_dropout = nn.Sequential(
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size, proxy_cnt, bn_type):
        return ProxyAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels, 
                                    size,
                                    proxy_cnt, bn_type)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class Proxy_ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36), bn_type=None):
        super(Proxy_ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(out_features, bn_type=bn_type),
                                     ProxyOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                              key_channels=out_features//2, value_channels=out_features//2,
                                                              dropout=0, sizes=([1]), proxy_cnt=128, bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(0.1)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
