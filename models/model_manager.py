#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for img segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.nets.asp_ocnet import AspOCNet
from models.nets.asp_ocnetv2 import AspOCNetV2
from models.nets.asp_ocnetv3 import AspOCNetV3
from models.nets.asp_ocnetv4 import AspOCNetV4
from models.nets.asp_ocnetv5 import AspOCNetV5
from models.nets.deeplabv3 import DeepLabV3
from models.nets.denseassp import DenseASPP
from models.nets.pspnet import PSPNet
from models.nets.pyramid_ocnet import PyramidOCNet
from models.nets.base_ocnet import BaseOCNet
from models.nets.base_ocnetv2 import BaseOCNetV2
from utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    'deeplabv3': DeepLabV3,
    'pspnet': PSPNet,
    'denseaspp': DenseASPP,
    'asp_ocnet': AspOCNet,
    'asp_ocnetv2': AspOCNetV2,
    'asp_ocnetv3': AspOCNetV3,
    'asp_ocnetv4': AspOCNetV4,
    'asp_ocnetv5': AspOCNetV5,
    'base_ocnet': BaseOCNet,
    'base_ocnetv2': BaseOCNetV2,
    'pyramid_ocnet': PyramidOCNet
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
