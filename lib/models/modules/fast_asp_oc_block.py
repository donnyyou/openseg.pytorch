##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/AlexHex7/Non-local_pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
from models.modules.fast_base_oc_block import FastBaseOC_Context_Module
from torch.autograd import Variable
from torch.nn import functional as F

from lib.models.tools import ModuleHelper


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


class Fast_ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None):
        super(Fast_ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type),
                                   FastBaseOC_Context_Module(in_channels=out_features,
                                                             key_channels=out_features//2, scale=1, bn_type=bn_type),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type),)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(0.1)
            )

        self.object_head = ObjectContext_Module(num_classes)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        feat1 = self.context[2](feat1, proxy_feats)
        # pdb.set_trace()
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
