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
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    from extensions.inplace_abn.bn import InPlaceABNSync

elif torch_ver == '0.3':
    from extensions.inplace_abn_03.modules import InPlaceABNSync

from models.modules.base_oc_block import BaseOC_Context_Module
from models.modules.fix_base_oc_block import Fix_BaseOC_Context_Module
from models.modules.proxy_oc_block import ProxyOC_Context_Module
from models.modules.oc_conv_block import OC_Conv_Module


class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   InPlaceABNSync(out_features),
                                   BaseOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features//2, 
                                    dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
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


class Fix_ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(Fix_ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   InPlaceABNSync(out_features),
                                   Fix_BaseOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features//2, 
                                    dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
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


class ASP_Proxy_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_Proxy_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   InPlaceABNSync(out_features),
                                   ProxyOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features//2, 
                                    dropout=0, sizes=([1]), proxy_cnt=128))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
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


class ASP_Proxy_OC_Module_v2(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_Proxy_OC_Module_v2, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   InPlaceABNSync(out_features),
                                   ProxyOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features//2, 
                                    dropout=0, sizes=([2]), proxy_cnt=128))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
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


class ASP_OC_Conv_Module(nn.Module):
    def __init__(self, features, dilations=(12, 24, 36)):
        super(ASP_OC_Conv_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   InPlaceABNSync(features),
                                   BaseOC_Context_Module(in_channels=features, out_channels=features, key_channels=features//2, value_channels=features//2, 
                                    dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(features))

        self.conv3 = nn.Sequential(OC_Conv_Module(features, features//2, features, features, kernel_size=3, 
                                                    dilation=dilations[0], padding=dilations[0]),
                                   InPlaceABNSync(features))
        self.conv4 = nn.Sequential(OC_Conv_Module(features, features//2, features, features, kernel_size=3, 
                                                    dilation=dilations[1], padding=dilations[1]),
                                   InPlaceABNSync(features))
        self.conv5 = nn.Sequential(OC_Conv_Module(features, features//2, features, features, kernel_size=3, 
                                                    dilation=dilations[2], padding=dilations[2]),
                                   InPlaceABNSync(features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(features * 5, features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(features),
            nn.Dropout2d(0.1)
            )
        

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
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