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
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class AspOCNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(AspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        from lib.models.modules.asp_oc_block import ASP_OC_Module
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            ASP_OC_Module(512, 512, bn_type=self.configer.get('network', 'bn_type')),
        )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.context.apply(self.weights_init)
        self.cls.apply(self.weights_init)
        self.dsn.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.context(x[-1])
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x
