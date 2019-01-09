#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class FSDataLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.name_list = self.__list_dirs(root_dir, dataset)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        if self.configer.exists('data', 'label_list'):
            labelmap = self._encode_label(labelmap)

        if self.configer.exists('data', 'reduce_zero_label'):
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.tonp(labelmap)
        ori_target[ori_target == 255] = -1

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap=labelmap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )
        return dict(
            img=DataContainer(img, stack=True),
            labelmap=DataContainer(labelmap, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=self.name_list[index],
        )

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(label_dir):
            image_name = file_name.split('.')[0]
            img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
            label_path = os.path.join(label_dir, file_name)
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')
            for file_name in os.listdir(label_dir):
                image_name = file_name.split('.')[0]
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        return img_list, label_list, name_list


class FSDataTestLoader(data.Dataset):
    def __init__(self, root_dir, dataset=None, img_transform=None, configer=None):
        self.configer = configer
        self.img_transform = img_transform
        self.img_list, self.name_list = self.__list_dirs(root_dir, dataset)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        if self.img_transform is not None:
            img = self.img_transform(img)
        meta = dict(
            ori_img_size=img_size,
        )
        return dict(
            img=DataContainer(img, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=self.name_list[index],
        )

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset)
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for item in os.listdir(image_dir):
            sub_image_dir = os.path.join(image_dir, item)
            for file_name in os.listdir(sub_image_dir):
                image_name = file_name.split('.')[0]
                img_path = os.path.join(sub_image_dir, file_name)
                if not os.path.exists(img_path):
                    Log.error('Image Path: {} not exists.'.format(img_path))
                    continue
                img_list.append(img_path)
                name_list.append(image_name)

        return img_list, name_list

if __name__ == "__main__":
    # Test cityscapes loader.
    pass
