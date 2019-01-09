#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pdb

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from lib.vis.seg_visualizer import SegVisualizer
from lib.datasets.seg_data_loader import SegDataLoader
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_parser import SegParser
from segmentor.tools.blob_helper import BlobHelper
from segmentor.tools.module_runner import ModuleRunner


class TesterMulGPU(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_parser = SegParser(configer)
        self.seg_model_manager = ModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)
        self.module_runner = ModuleRunner(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.seg_net = None
        self.test_size = None
        self.save_dir = self.configer.get('test', 'out_dir')
        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)
        if 'test' in self.save_dir:
            self.test_loader = self.seg_data_loader.get_testloader()
            self.test_size = len(self.test_loader) * self.configer.get('test', 'batch_size')
        else:
            self.test_loader = self.seg_data_loader.get_valloader()
            self.test_size = len(self.test_loader) * self.configer.get('val', 'batch_size')

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('data', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst

    def test(self):
        self.seg_net.eval()
        start_time = time.time()
        image_id = 0
        Log.info('save dir {}'.format(self.save_dir))
        FileHelper.make_dirs(self.save_dir, is_file=False)
        with torch.no_grad():
            for i, data_dict in enumerate(self.test_loader):
                inputs = data_dict['img']
                names = data_dict['name']
                outputs = self.seg_net.forward(inputs)
                outputs = outputs[-1]

                outputs = F.interpolate(outputs, size=(inputs.size(2), inputs.size(3)), mode='bilinear', align_corners=True)
                outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()

                for k in range(inputs.size(0)):
                    image_id += 1
                    label_img = np.asarray(np.argmax(outputs[k], axis=-1), dtype=np.uint8)
                    if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label'):
                        label_img = label_img + 1
                        label_img = label_img.astype(np.uint8)
                    label_img = self.__relabel(label_img)
                    label_img = Image.fromarray(label_img, 'P')
                    Log.info('{:4d}/{:4d} label map generated'.format(image_id, self.test_size))
                    label_path = os.path.join(self.save_dir, '{}.png'.format(names[k]))
                    ImageHelper.save(label_img, label_path)
