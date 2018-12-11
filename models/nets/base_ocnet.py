##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector

torch_ver = torch.__version__[:3]


class BaseOCNet(nn.Module):
    def __init__(self, configer):
        if torch_ver == '0.4':
            from extensions.inplace_abn.bn import InPlaceABNSync
        elif torch_ver == '0.3':
            from extensions.inplace_abn_03.modules import InPlaceABNSync

        self.inplanes = 128
        super(BaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            )
        from models.modules.base_oc_block import BaseOC_Module
        self.oc_module = BaseOC_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256,
            dropout=0.05, sizes=([1]))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.oc_module_pre(x[-1])
        x = self.oc_module(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=False)
        return aux_x, x

