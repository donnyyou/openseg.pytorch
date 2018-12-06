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

from models.modules.oc_modules.proxy_oc_block import ProxyOC_Context_Module
from models.backbones.backbone_selector import BackboneSelector

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync


class ResNet101ProxyOCContextV5(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(ResNet101ProxyOCContextV5, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.context4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                dropout=0, sizes=([1]), proxy_cnt=128)
            )
        self.context3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                dropout=0, sizes=([1]), proxy_cnt=128)
            )
        self.context2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                dropout=0, sizes=([1]), proxy_cnt=128)
            )

        self.context_fuse = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            )

        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        context2 = self.context2(x)
        x = self.layer3(x)
        context3 = self.context3(x)
        x_dsn = self.dsn(context3)
        x = self.layer4(x)
        context4 = self.context4(x)
        x = self.context_fuse(torch.cat([context2, context3, context4], 1))
        x = self.cls(x)
        return [x_dsn, x]

