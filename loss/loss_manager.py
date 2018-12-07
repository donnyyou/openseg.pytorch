#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss Manager for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.seg_modules import FSCELoss, FSAuxCELoss, FSAuxEncCELoss
from utils.tools.logger import Logger as Log



SEG_LOSS_DICT = {
    'fs_ce_loss': FSCELoss,
    'fs_auxce_loss':FSAuxCELoss,
    'fs_auxencce_loss': FSAuxEncCELoss
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            from extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_seg_loss(self):
        key = self.configer.get('loss', 'loss_type')
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)


