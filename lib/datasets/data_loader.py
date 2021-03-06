#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Class for the Semantic Segmentation Data Loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils import data

import lib.datasets.tools.transforms as trans
import lib.datasets.tools.cv2_aug_transforms as cv2_aug_trans
import lib.datasets.tools.pil_aug_transforms as pil_aug_trans
from lib.datasets.loader.default_loader import DefaultLoader
from lib.datasets.loader.ade20k_loader import ADE20KLoader
from lib.datasets.tools.collate import collate
from lib.utils.tools.logger import Logger as Log


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])

    def get_trainloader(self):
        if self.configer.exists('train', 'loader') and self.configer.get('train', 'loader') == 'ade20k':
            trainloader = data.DataLoader(
                ADE20KLoader(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             label_transform=self.label_transform,
                             configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'),
                shuffle=True, drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

        else:
            trainloader = data.DataLoader(
                DefaultLoader(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                              aug_transform=self.aug_train_transform,
                              img_transform=self.img_transform,
                              label_transform=self.label_transform,
                              configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'),
                shuffle=True, drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset
        if self.configer.get('method') == 'fcn_segmentor':
            valloader = data.DataLoader(
                DefaultLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                              aug_transform=self.aug_val_transform,
                              img_transform=self.img_transform,
                              label_transform=self.label_transform,
                              configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'), shuffle=False,
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('val', 'data_transformer')
                )
            )

            return valloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None


if __name__ == "__main__":
    # Test data loader.
    pass
