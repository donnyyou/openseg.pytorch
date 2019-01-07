#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for img segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.nets.asp_ocnet import AspOCNet
from lib.models.nets.base_ocnet import BaseOCNet
from lib.models.nets.deeplabv3 import DeepLabV3
from lib.models.nets.denseassp import DenseASPP
from lib.models.nets.fast_asp_ocnet import FastAspOCNet
from lib.models.nets.fast_base_ocnet import FastBaseOCNet
from lib.models.nets.proxy_base_ocnet import ProxyBaseOCNet
from lib.models.nets.proxy_asp_ocnet import ProxyAspOCNet
from lib.models.nets.pspnet import PSPNet
from lib.models.nets.pyramid_ocnet import PyramidOCNet
from lib.utils.tools.logger import Logger as Log

from lib.models.nets.rainbow_fast_base_ocnet import RainbowFastBaseOCNet

SEG_MODEL_DICT = {
    'deeplabv3': DeepLabV3,
    'pspnet': PSPNet,
    'denseaspp': DenseASPP,
    'asp_ocnet': AspOCNet,
    'base_ocnet': BaseOCNet,
    'pyramid_ocnet': PyramidOCNet,
    'fast_base_ocnet': FastBaseOCNet,
    'rainbow_fast_base_ocnet': RainbowFastBaseOCNet,
    'fast_asp_ocnet': FastAspOCNet,
    'proxy_base_ocnet': ProxyBaseOCNet,
    'proxy_asp_ocnet': ProxyAspOCNet,
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
