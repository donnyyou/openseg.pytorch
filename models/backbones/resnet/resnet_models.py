#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch.nn as nn
from collections import OrderedDict

from models.tools.module_helper import ModuleHelper
from utils.tools.logger import Logger as Log


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False, bn_type=None):
        super(ResNet, self).__init__()
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)),
                ('relu3', nn.ReLU(inplace=False))]
            ))
        else:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)),
                ('relu1', nn.ReLU(inplace=False))]
            ))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], bn_type=bn_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_type=bn_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_type=bn_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_type=bn_type)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, ModuleHelper.BatchNorm2d(bn_type=bn_type, ret_cls=True)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, bn_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bn_type=bn_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_type=bn_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetModels(object):

    def __init__(self, configer):
        self.configer = configer

    def resnet18(self, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], deep_base=False,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def deepbase_resnet18(self, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], deep_base=False,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def deepbase_resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=False,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def deepbase_resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=False,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def deepbase_resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def resnet152(self, **kwargs):
        """Constructs a ResNet-152 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=False,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def deepbase_resnet152(self, **kwargs):
        """Constructs a ResNet-152 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model
