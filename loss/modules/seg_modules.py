#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -100
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSEdgeLoss(nn.Module):
    def __init__(self, configer=None):
        super(FSEdgeLoss, self).__init__()
        self.configer = configer

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._get_edgemap(targets[0], (inputs.size(2), inputs.size(3)))
            edge_sum = target.sum()
            total_cnt = inputs.size(2) * inputs.size(3)
            weights = torch.FloatTensor([total_cnt / (total_cnt - edge_sum), total_cnt / edge_sum]).cuda()
            loss = F.cross_entropy(inputs, target, weight=weights)
        return loss

    @staticmethod
    def _get_edgemap(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=[x * 1 for x in scaled_size], mode='nearest')
        neg_targets = -targets
        edgemap = F.max_pool2d(targets, 3, 1, 1) + F.max_pool2d(neg_targets, 3, 1, 1)
        edgemap = F.interpolate(edgemap, size=scaled_size, mode='nearest')
        edgemap[edgemap != 0] = 1
        return edgemap.squeeze(1).long()


class FSEdgeAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSEdgeAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.edge_loss = FSEdgeLoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        edge_out, aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
        aux_loss = self.ce_loss(aux_out, aux_targets)
        edge_loss = self.edge_loss(edge_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        loss = loss + self.configer.get('network', 'loss_weights')['edge_loss'] * edge_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


if __name__ == "__main__":
    inputs = torch.ones((3, 5, 6, 6)).cuda()
    targets = torch.ones((3, 6, 6)).cuda()
