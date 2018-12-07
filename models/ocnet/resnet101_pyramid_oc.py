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
import torch

from models.modules.oc_modules.pyramid_oc_block import Pyramid_OC_Module
from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper

torch_ver = torch.__version__[:3]


class ResNet101PyramidOC(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(ResNet101PyramidOC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get('network', 'bn_type'))(512)
        )

        # extra added layers
        self.context = Pyramid_OC_Module(in_channels=512, out_channels=512, dropout=0.05, sizes=([1, 2, 3, 6]))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get('network', 'bn_type'))(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        x = self.backbone(x)
        x_dsn = self.dsn(x[-2])
        x = self.layer5(x[-1])
        x = self.context(x)
        x = self.cls(x)
        return [x_dsn, x] 

