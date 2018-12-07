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

from models.backbones.backbone_selector import BackboneSelector


torch_ver = torch.__version__[:3]


class AspOCNet(nn.Module):
    def __init__(self, configer):
        if torch_ver == '0.4':
            from extensions.inplace_abn.bn import InPlaceABNSync
        elif torch_ver == '0.3':
            from extensions.inplace_abn_03.modules import InPlaceABNSync

        self.inplanes = 128
        super(AspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        from models.modules.oc_modules.asp_oc_block import ASP_OC_Module
        self.context = nn.Sequential(
                ASP_OC_Module(2048, 512),
                )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        x = self.backbone(x)
        aux_x = self.dsn(x[-1])
        x = self.layer4(x)
        x = self.context(x)
        x = self.cls(x)
        return aux_x, x
