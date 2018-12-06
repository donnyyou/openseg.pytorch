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
from torch.nn import functional as F
import torch

from models.modules.oc_modules.proxy_oc_block import ProxyOC_Context_Module
from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync


class Decoder_Module(nn.Module):
    '''
    Reference: Liang-Chieh Chen etc.  "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
               use a decoder to refine the object boundaries.
    '''
    def __init__(self, num_classes, bn_type):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(256),
            nn.ReLU(inplace=False)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(48),
            nn.ReLU(inplace=False)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(256),
            nn.ReLU(inplace=False)
            )
        self.fuse_context = ProxyOC_Context_Module(in_channels=256, out_channels=256, key_channels=128, value_channels=128, 
            dropout=0.05, sizes=([1]), proxy_cnt=128)       
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.upsample(self.conv1(xt), size=(h, w), mode='bilinear')
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        x = self.fuse_context(x)
        seg = self.conv4(x)
        return seg


class ResNet101ProxyOCContextV6(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(ResNet101ProxyOCContextV6, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            )
        # self.context = ProxyOC_Context_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
        #     dropout=0.05, sizes=([1]), proxy_cnt=128)
        self.decoder = Decoder_Module(self.num_classes, self.configer.get('network', 'bn_type'))
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        x = self.backbone(x)
        x_dsn = self.dsn(x[-2])
        x = self.layer5(x[-1])
        # x = self.context(x)
        x = self.decoder(x, x[-4])
        return [x_dsn, x]

