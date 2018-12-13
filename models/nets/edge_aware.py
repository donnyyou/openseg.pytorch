#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pytorch implementation of PSP net Synchronized Batch Normalization
# this is pytorch implementation of PSP resnet101 (syn-bn) version


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1, relu=True, bn_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation = dilation, bias=False)
        self.bn = ModuleHelper.BatchNorm2d(bn_type=bn_type)(num_features=outplanes)
        self.relu_f = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu:
            x = self.relu_f(x)
        return x


# PSP decoder Part
# pyramid pooling, bilinear upsample
class EdgeModule(nn.Module):
    def __init__(self, out_planes=256, bn_type=None):
        super(EdgeModule, self).__init__()
        self.bn_type = bn_type
        self.edge_conv1 = _ConvBatchNormReluBlock(512, out_planes, 1, 1, padding=0, bn_type=bn_type)
        self.edge_conv2 = _ConvBatchNormReluBlock(1024, out_planes, 1, 1, padding=0, bn_type=bn_type)

    def forward(self, x1, x2):
        out1 = self.edge_conv1(x1)
        out2 = self.edge_conv2(x2)
        ppm_out = torch.cat([out1, out2], 1)
        return ppm_out


class EdgeAwareNet(nn.Sequential):
    def __init__(self, configer):
        super(EdgeAwareNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        num_features = self.backbone.get_num_features()
        self.dsn = nn.Sequential(
            _ConvBatchNormReluBlock(num_features // 2, num_features // 4, 3, 1,
                                    bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(num_features // 4, self.num_classes, 1, 1, 0)
        )
        self.edge_module = EdgeModule(out_planes=256, bn_type=self.configer.get('network', 'bn_type'))
        self.edge_cls = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(num_features + 512, 512, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get('network', 'bn_type'))(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux = self.dsn(x[-2])
        edge_f = self.edge_module(x[-3], x[-2])
        edge_x = self.edge_cls(edge_f)
        x = torch.cat([x[-1], edge_f], 1)
        x = self.cls(x)
        edge_x = F.interpolate(edge_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return edge_x, aux, x


if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    model = EdgeAwareNet(num_classes=19).cuda()
    model.eval()
    o, _ = model(i)
    #print(o.size())
    #final_out = F.upsample(o,scale_factor=8)
    #print(final_out.size())
    print(o.size())
    print(_.size())