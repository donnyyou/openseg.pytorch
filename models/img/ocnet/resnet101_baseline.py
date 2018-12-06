##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## updated by: RainbowSecret
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

affine_par = True
torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync


class ResNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.layer5 = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
                InPlaceABNSync(512),
                nn.Dropout2d(0.05)
                )
        self.layer6 = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
