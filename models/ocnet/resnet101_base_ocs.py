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

from models.modules.oc_modules.base_ocs_block import BaseOCS_Module
from models.backbones.backbone_selector import BackboneSelector

affine_par = True
torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync


class ResNet101BaseOCsDsn(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(ResNet101BaseOCsDsn, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            BaseOCS_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([1]))
            )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.inter_cls = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        x = self.backbone(x)
        x_dsn = self.dsn(x[-2])
        x, inter_map = self.context(x[-1])
        x_inter = self.inter_cls(inter_map)
        x = self.cls(x)
        return [x_dsn, x_inter, x]
