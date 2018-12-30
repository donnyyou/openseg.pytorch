#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for img segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.nets.asp_ocnetv4 import AspOCNetV4
from models.nets.fast_asp_ocnet import FastAspOCNet
from models.nets.deeplabv3 import DeepLabV3
from models.nets.denseassp import DenseASPP
from models.nets.pspnet import PSPNet
from models.nets.pyramid_ocnet import PyramidOCNet
from models.nets.base_ocnet import BaseOCNet
from models.nets.fast_base_ocnet import FastBaseOCNet
from utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    'deeplabv3': DeepLabV3,
    'pspnet': PSPNet,
    'denseaspp': DenseASPP,
    'asp_ocnetv4': AspOCNetV4,
    'base_ocnet': BaseOCNet,
    'pyramid_ocnet': PyramidOCNet,
    'fast_base_ocnet': FastBaseOCNet,
    'fast_asp_ocnet': FastAspOCNet
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
