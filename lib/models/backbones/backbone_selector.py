#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from libmodels.backbones.mobilenet.mobilenet_backbone import MobileNetBackbone

from lib.models.backbones.densenet.densenet_backbone import DenseNetBackbone
from lib.models.backbones.resnet.resnet_backbone import ResNetBackbone
from lib.models.backbones.squeezenet.squeezenet_backbone import SqueezeNetBackbone
from lib.utils.tools.logger import Logger as Log


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone')

        model = None
        if 'resnet' in backbone:
            model = ResNetBackbone(self.configer)(**params)

        elif 'mobilenet' in backbone:
            model = MobileNetBackbone(self.configer)(*params)

        elif 'densenet' in backbone:
            model = DenseNetBackbone(self.configer)(**params)

        elif 'squeezenet' in backbone:
            model = SqueezeNetBackbone(self.configer)(**params)

        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model
