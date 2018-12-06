##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/KaiyuYue/cgnl-network.pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import os
import sys
import pdb
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
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


class SpatialCGNLx(nn.Module):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=None, order=2):
        self.use_scale = use_scale
        self.groups = groups
        self.order = order

        super(SpatialCGNLx, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNLx block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNLx block uses '{}' groups".format(self.groups), \
                   'yellow')

        print('=> WARN: The Taylor expansion order in SpatialCGNLx block is {}'.format(self.order), \
               'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The non-linear kernel (Gaussian RBF).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """

        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        # gamma
        gamma = torch.Tensor(1).fill_(1e-4)

        # beta
        beta = torch.exp(-2 * gamma)

        t_taylor = []
        p_taylor = []
        for order in range(self.order+1):
            # alpha
            alpha = torch.mul(
                    torch.div(
                    torch.pow(
                        (2 * gamma),
                        order),
                        math.factorial(order)),
                        beta)

            alpha = torch.sqrt(
                        alpha.cuda())

            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)

            t_taylor.append(_t)
            p_taylor.append(_p)

        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)

        att = torch.bmm(p_taylor, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        att = att.view(b, 1, int(self.order+1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x
