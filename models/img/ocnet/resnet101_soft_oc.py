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

from models.modules.utils.psp_block import GC_Module
from models.modules.oc_modules.soft_oc_block import SoftOC_Module
from models.backbones.backbone_selector import BackboneSelector

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync


class ResNet101SoftOC(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(ResNet101SoftOC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            )
        self.global_context = GC_Module(512, 512)
        self.object_context = SoftOC_Module(cls_num=self.num_classes)

        self.cls_global = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            InPlaceABNSync(512),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            )
        self.cls_object_global = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, padding=0),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            )

    def forward(self, x):
        x = self.backbone(x)
        feats = self.layer5(x[-1])
        global_context = self.global_context(feats)
        global_cls = self.cls_global(torch.cat([feats, global_context], 1))
        object_context = self.object_context(feats, global_cls)
        object_global_cls = self.cls_object_global(torch.cat([feats, global_context, object_context], 1))
        return [global_cls, object_global_cls]
