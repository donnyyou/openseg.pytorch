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


class DPC_Module(nn.Module):
    """
    Reference: 
        Searching for Efficient Multi-Scale Architectures for Dense Image Prediction .
    """
    def __init__(self, features, out_features):
        super(DPC_Module, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(1, 6), dilation=(1, 6), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(18, 15), dilation=(18, 15), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(6, 21), dilation=(6, 21), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(1, 1), dilation=(1, 1), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(6, 3), dilation=(6, 3), bias=False),
                                   InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat1)
        feat4 = self.conv4(feat1)
        feat5 = self.conv5(feat2)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        out = self.conv_bn_dropout(out)
        return out


class DPC_v2_Module(nn.Module):
    """
    Reference: 
        mIoU on val set = 78.16
        Searching for Efficient Multi-Scale Architectures for Dense Image Prediction .
    """
    def __init__(self, features, out_features):
        super(DPC_v2_Module, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(6, 1), dilation=(6, 1), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(21, 15), dilation=(21, 15), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(12, 21), dilation=(12, 21), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(3, 6), dilation=(3, 6), bias=False),
                                   InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(x)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        out = self.conv_bn_dropout(out)
        return out


class DPC_v3_Module(nn.Module):
    """
    Reference: 
        mIoU on val set = 78.16
        Searching for Efficient Multi-Scale Architectures for Dense Image Prediction .
    """
    def __init__(self, features, out_features):
        super(DPC_v3_Module, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(12, 1), dilation=(12, 1), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(1, 6), dilation=(1, 6), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=(21, 21), dilation=(21, 21), bias=False),
                                   InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat1)
        feat4 = self.conv4(feat2)
        feat5 = self.conv5(x)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        out = self.conv_bn_dropout(out)
        return out