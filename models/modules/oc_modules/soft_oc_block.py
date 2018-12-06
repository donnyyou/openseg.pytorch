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
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
    from bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    
elif torch_ver == '0.3':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn_03'))
    from modules import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')  


class SoftOC_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=19):
        super(SoftOC_Module, self).__init__()
        self.cls_num = cls_num

    def forward(self, feats, probs):
        probs = F.softmax(probs, dim=1)
        n, c, h, w = feats.size()
        soft_context = nn.Parameter(torch.zeros(n, c, h, w).type(torch.FloatTensor).cuda(), requires_grad=False)
        for cls_id in range(self.cls_num):
            _prob = probs[:, cls_id, :, :]
            _prob_extend = torch.unsqueeze(_prob,1)
            _class_context = torch.sum(torch.sum(torch.mul(feats, _prob_extend), 2), 2) / torch.sum(_prob)
            soft_context += torch.mul(F.upsample(input=_class_context.unsqueeze(2).unsqueeze(3), size=(h, w), mode='bilinear'), _prob_extend)
        return soft_context


class HardOC_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
    """
    def __init__(self, features, out_features=512, dropout=0.1, cls_num=19, thres=0.5):
        super(HardOC_Module, self).__init__()
        self.cls_num = cls_num
        self.thres = thres
        self.relu = nn.ReLU(inplace=True)
        self.conv_bn_drop = nn.Sequential(
            nn.Conv2d(2*features, out_features, kernel_size=1, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(dropout)
            )

    def forward(self, feats, probs):
        probs = F.softmax(probs, dim=1)
        n, c, h, w = feats.size()
        context = nn.Parameter(torch.zeros(n, c, h, w).type(torch.FloatTensor).cuda(), requires_grad=False)
        _max_prob, _ = torch.max(probs, 1)
        for cls_id in range(self.cls_num):
            _prob = probs[:, cls_id, :, :]
            # Hard Context Feature compute the context features based masking
            _mask = torch.eq(_prob, _max_prob)
            _mask = _mask.type(torch.cuda.FloatTensor)
            _mask_extend = torch.unsqueeze(_mask,1).repeat(1,c,1,1)
            _mean = torch.sum(torch.sum(torch.mul(feats, _mask_extend), 2), 2) / torch.sum(_mask)
            context += torch.mul(F.upsample(input=_mean.unsqueeze(2).unsqueeze(3),  size=(h, w), mode='bilinear'), _mask_extend)

        out = self.conv_bn_drop(torch.cat([feats, context], 1))
        return out
