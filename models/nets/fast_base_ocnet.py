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
from torch.nn import functional as F

from models.modules.fast_base_oc_block import FastBaseOC_Module
from models.tools.module_helper import ModuleHelper
from models.backbones.backbone_selector import BackboneSelector


class ObjectContext_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0):
        super(ObjectContext_Module, self).__init__()
        self.cls_num = cls_num

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(probs, dim=2)# batch x c x hw
        cc = torch.matmul(probs, feats)# batch x c x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class FastBaseOCNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.down = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            )
        self.object_head = ObjectContext_Module(self.num_classes)
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, scale=1, dropout=0.05)
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.down(x[-1])
        context = self.object_head(x, x_dsn)
        x = self.fast_oc_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x

