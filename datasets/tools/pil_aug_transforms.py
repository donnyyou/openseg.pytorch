#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import math
import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from utils.tools.logger import Logger as Log


class RandomPad(object):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.

            Returns::
                img: Image object.
    """
    def __init__(self, up_scale_range=None, pad_ratio=0.5, mean=(104, 117, 123)):
        # do something
        assert isinstance(up_scale_range, (list, tuple))
        self.up_scale_range = up_scale_range
        self.ratio = pad_ratio
        self.mean = tuple(mean)

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        width, height = img.size
        ws = random.uniform(self.up_scale_range[0], self.up_scale_range[1])
        hs = ws
        for _ in range(50):
            scale = random.uniform(self.up_scale_range[0], self.up_scale_range[1])
            min_ratio = max(0.5, 1. / scale / scale)
            max_ratio = min(2, scale * scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            ws = scale * ratio
            hs = scale / ratio
            if ws >= 1 and hs >= 1:
                break

        w = int(ws * width)
        h = int(hs * height)

        pad_width = random.randint(0, w - width)
        pad_height = random.randint(0, h - height)

        left_pad = random.randint(0, pad_width)  # pad_left
        up_pad = random.randint(0, pad_height)  # pad_up
        right_pad = pad_width - left_pad  # pad_right
        down_pad = pad_height - up_pad  # pad_down

        img = ImageOps.expand(img, (left_pad, up_pad, right_pad, down_pad), fill=self.mean)

        if labelmap is not None:
            labelmap = ImageOps.expand(labelmap, (left_pad, up_pad, right_pad, down_pad), fill=255)

        if maskmap is not None:
            maskmap = ImageOps.expand(maskmap, (left_pad, up_pad, right_pad, down_pad), fill=1)

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] += left_pad
                    polygons[object_id][polygon_id][1::2] += up_pad

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] += left_pad
            kpts[:, :, 1] += up_pad

        if bboxes is not None and bboxes.size > 0:
            bboxes[:, 0::2] += left_pad
            bboxes[:, 1::2] += up_pad

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class Padding(object):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.
            Returns::
                img: Image object.
    """
    def __init__(self, pad=None, pad_ratio=0.5, mean=(104, 117, 123), allow_outside_center=True):
        self.pad = pad
        self.ratio = pad_ratio
        self.mean = tuple(mean)
        self.allow_outside_center = allow_outside_center

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        width, height = img.size
        left_pad, up_pad, right_pad, down_pad = self.pad
        target_size = [width + left_pad + right_pad, height + up_pad + down_pad]
        offset_left = -left_pad
        offset_up = -up_pad

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] -= offset_left
            kpts[:, :, 1] -= offset_up
            mask = np.logical_or.reduce((kpts[:, :, 0] >= target_size[0], kpts[:, :, 0] < 0,
                                         kpts[:, :, 1] >= target_size[1], kpts[:, :, 1] < 0))
            kpts[mask == 1, 2] = -1

        if bboxes is not None and bboxes.size > 0:
            if self.allow_outside_center:
                mask = np.ones(bboxes.shape[0], dtype=bool)
            else:
                crop_bb = np.array([offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]])
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

            bboxes[:, 0::2] -= offset_left
            bboxes[:, 1::2] -= offset_up
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, target_size[0] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, target_size[1] - 1)

            mask = np.logical_and(mask, (bboxes[:, :2] < bboxes[:, 2:]).all(axis=1))
            bboxes = bboxes[mask]
            if labels is not None:
                labels = labels[mask]

            if polygons is not None:
                new_polygons = list()
                for object_id in range(len(polygons)):
                    if mask[object_id] == 1:
                        for polygon_id in range(len(polygons[object_id])):
                            polygons[object_id][polygon_id][0::2] -= offset_left
                            polygons[object_id][polygon_id][1::2] -= offset_up
                            polygons[object_id][polygon_id][0::2] = np.clip(polygons[object_id][polygon_id][0::2],
                                                                            0, target_size[0] - 1)
                            polygons[object_id][polygon_id][1::2] = np.clip(polygons[object_id][polygon_id][1::2],
                                                                            0, target_size[1] - 1)

                        new_polygons.append(polygons[object_id])

                polygons = new_polygons

        img = ImageOps.expand(img, border=tuple(self.pad), fill=tuple(self.mean))
        if maskmap is not None:
            maskmap = ImageOps.expand(maskmap, border=tuple(self.pad), fill=1)

        if labelmap is not None:
            labelmap = ImageOps.expand(labelmap, border=tuple(self.pad), fill=255)

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomHFlip(object):
    def __init__(self, swap_pair=None, flip_ratio=0.5):
        self.swap_pair = swap_pair
        self.ratio = flip_ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        width, height = img.size
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if labelmap is not None:
            labelmap = labelmap.transpose(Image.FLIP_LEFT_RIGHT)

        if maskmap is not None:
            maskmap = maskmap.transpose(Image.FLIP_LEFT_RIGHT)

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] = width - 1 - polygons[object_id][polygon_id][0::2]

        if bboxes is not None and bboxes.size > 0:
            xmin = width - 1 - bboxes[:, 2]
            xmax = width - 1 - bboxes[:, 0]
            bboxes[:, 0] = xmin
            bboxes[:, 2] = xmax

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] = width - 1 - kpts[:, :, 0]

            for pair in self.swap_pair:
                temp_point = np.copy(kpts[:, pair[0] - 1])
                kpts[:, pair[0] - 1] = kpts[:, pair[1] - 1]
                kpts[:, pair[1] - 1] = temp_point

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, saturation_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = saturation_ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8)), labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomHue(object):
    def __init__(self, delta=18, hue_ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = hue_ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8)), labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomPerm(object):
    def __init__(self, perm_ratio=0.5):
        self.ratio = perm_ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        swap = self.perms[random.randint(0, len(self.perms)-1)]
        img = np.array(img)
        img = img[:, :, swap]
        return Image.fromarray(img.astype(np.uint8)), labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, contrast_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = contrast_ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = np.array(img).astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255)

        return Image.fromarray(img.astype(np.uint8)), labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomBrightness(object):
    def __init__(self, shift_value=30, brightness_ratio=0.5):
        self.shift_value = shift_value
        self.ratio = brightness_ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        shift = np.random.uniform(-self.shift_value, self.shift_value, size=1)
        image = np.array(img).astype(np.float32)
        image[:, :, :] += shift
        image = np.around(image)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)

        return image, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomGaussBlur(object):
    def __init__(self, max_blur=4, blur_ratio=0.5):
        self.max_blur = max_blur
        self.ratio = blur_ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        blur_value = np.random.uniform(0, self.max_blur)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_value))
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
    """

    def __init__(self, h_range, s_range, v_range, hsv_ratio=0.5):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
        self.ratio = hsv_ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = np.array(img)
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v * v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return Image.fromarray(img_new.astype(np.uint8)), labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomResize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_range=(0.75, 1.25), aspect_range=(0.9, 1.1), target_size=None,
                 resize_bound=None, method='random', max_side_bound=None, resize_ratio=0.5):
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.resize_bound = resize_bound
        self.max_side_bound = max_side_bound
        self.method = method
        self.ratio = resize_ratio

        if target_size is not None:
            if isinstance(target_size, int):
                self.input_size = (target_size, target_size)
            elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                self.input_size = target_size
            else:
                raise TypeError('Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size, bboxes):
        if self.method == 'random':
            scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'focus':
            if self.input_size is not None and bboxes is not None and len(bboxes) > 0:
                bboxes = np.array(bboxes)
                border = bboxes[:, 2:] - bboxes[:, 0:2]
                scale = 0.6 / max(max(border[:, 0]) / self.input_size[0], max(border[:, 1]) / self.input_size[1])
                scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1]) * scale
                return scale_ratio

            else:
                scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
                return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            Log.error('Resize method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img     (Image):   Image to be resized.
            maskmap    (Image):   Mask to be resized.
            kpt     (list):    keypoints to be resized.
            center: (list):    center points to be resized.

        Returns:
            Image:  Randomly resize image.
            Image:  Randomly resize maskmap.
            list:   Randomly resize keypoints.
            list:   Randomly resize center points.
        """
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        width, height = img.size
        if random.random() < self.ratio:
            scale_ratio = self.get_scale([width, height], bboxes)
            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
            if self.max_side_bound is not None and max(height*h_scale_ratio, width*w_scale_ratio) > self.max_side_bound:
                d_ratio = self.max_side_bound / max(height * h_scale_ratio, width * w_scale_ratio)
                w_scale_ratio *= d_ratio
                h_scale_ratio *= d_ratio
        else:
            w_scale_ratio, h_scale_ratio = 1.0, 1.0

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] *= w_scale_ratio
            kpts[:, :, 1] *= h_scale_ratio

        if bboxes is not None and bboxes.size > 0:
            bboxes[:, 0::2] *= w_scale_ratio
            bboxes[:, 1::2] *= h_scale_ratio

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] *= w_scale_ratio
                    polygons[object_id][polygon_id][1::2] *= h_scale_ratio

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))

        img = img.resize(converted_size, Image.BILINEAR)
        if labelmap is not None:
            labelmap = labelmap.resize(converted_size, Image.NEAREST)
        if maskmap is not None:
            maskmap = maskmap.resize(converted_size, Image.NEAREST)

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree, rotate_ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree
        self.ratio = rotate_ratio
        self.mean = tuple(mean)

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img    (Image):     Image to be rotated.
            maskmap   (Image):     Mask to be rotated.
            kpt    (np.array):      Keypoints to be rotated.
            center (list):      Center points to be rotated.

        Returns:
            Image:     Rotated image.
            list:      Rotated key points.
        """
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() < self.ratio:
            rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        else:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = np.array(img)
        height, width, _ = img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=self.mean)
        img = Image.fromarray(img.astype(np.uint8))
        if labelmap is not None:
            labelmap = np.array(labelmap)
            labelmap = cv2.warpAffine(labelmap, rotate_mat, (new_width, new_height),
                                      borderValue=(255, 255, 255), flags=cv2.INTER_NEAREST)
            labelmap = Image.fromarray(labelmap.astype(np.uint8))

        if maskmap is not None:
            maskmap = np.array(maskmap)
            maskmap = cv2.warpAffine(maskmap, rotate_mat, (new_width, new_height),
                                     borderValue=(1, 1, 1), flags=cv2.INTER_NEAREST)
            maskmap = Image.fromarray(maskmap.astype(np.uint8))

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    for i in range(len(polygons[object_id][polygon_id]) // 2):
                        x = polygons[object_id][polygon_id][i * 2]
                        y = polygons[object_id][polygon_id][i * 2 + 1]
                        p = np.array([x, y, 1])
                        p = rotate_mat.dot(p)
                        polygons[object_id][polygon_id][i * 2] = p[0]
                        polygons[object_id][polygon_id][i * 2 + 1] = p[1]

        if kpts is not None and kpts.size > 0:
            num_objects = len(kpts)
            num_keypoints = len(kpts[0])
            for i in range(num_objects):
                for j in range(num_keypoints):
                    x = kpts[i][j][0]
                    y = kpts[i][j][1]
                    p = np.array([x, y, 1])
                    p = rotate_mat.dot(p)
                    kpts[i][j][0] = p[0]
                    kpts[i][j][1] = p[1]

        # It is not right for object detection tasks.
        if bboxes is not None and bboxes.size > 0:
            for i in range(len(bboxes)):
                bbox_temp = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][1],
                             bboxes[i][0], bboxes[i][3], bboxes[i][2], bboxes[i][3]]

                for node in range(4):
                    x = bbox_temp[node * 2]
                    y = bbox_temp[node * 2 + 1]
                    p = np.array([x, y, 1])
                    p = rotate_mat.dot(p)
                    bbox_temp[node * 2] = p[0]
                    bbox_temp[node * 2 + 1] = p[1]

                bboxes[i] = [min(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                             min(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                             max(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                             max(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7])]

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, crop_ratio=0.5, method='focus', grid=None, allow_outside_center=True):
        self.ratio = crop_ratio
        self.method = method
        self.grid = grid
        self.allow_outside_center = allow_outside_center

        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_lefttop(self, crop_size, img_size):
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]

        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]

        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]

        else:
            Log.error('Crop method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img (Image):   Image to be cropped.
            maskmap (Image):  Mask to be cropped.
            kpts (np.array):    keypoints to be cropped.
            bboxes (np.array): bounding boxes.

        Returns:
            Image:  Cropped image.
            Image:  Cropped maskmap.
            np.array:   Cropped keypoints.
            np.ndarray:   Cropped center points.
        """
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        target_size = (min(self.size[0], img.size[0]), min(self.size[1], img.size[1]))

        offset_left, offset_up = self.get_lefttop(target_size, img.size)

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] -= offset_left
            kpts[:, :, 1] -= offset_up

        if bboxes is not None and bboxes.size > 0:
            if self.allow_outside_center:
                mask = np.ones(bboxes.shape[0], dtype=bool)
            else:
                crop_bb = np.array([offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]])
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

            bboxes[:, 0::2] -= offset_left
            bboxes[:, 1::2] -= offset_up
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, target_size[0] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, target_size[1] - 1)

            mask = np.logical_and(mask, (bboxes[:, :2] < bboxes[:, 2:]).all(axis=1))
            bboxes = bboxes[mask]
            if labels is not None:
                labels = labels[mask]

            if polygons is not None:
                new_polygons = list()
                for object_id in range(len(polygons)):
                    if mask[object_id] == 1:
                        for polygon_id in range(len(polygons[object_id])):
                            polygons[object_id][polygon_id][0::2] -= offset_left
                            polygons[object_id][polygon_id][1::2] -= offset_up
                            polygons[object_id][polygon_id][0::2] = np.clip(polygons[object_id][polygon_id][0::2],
                                                                            0, target_size[0] - 1)
                            polygons[object_id][polygon_id][1::2] = np.clip(polygons[object_id][polygon_id][1::2],
                                                                            0, target_size[1] - 1)

                        new_polygons.append(polygons[object_id])

                polygons = new_polygons

        img = img.crop((offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]))

        if maskmap is not None:
            maskmap = maskmap.crop((offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]))

        if labelmap is not None:
            labelmap = labelmap.crop((offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]))

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class Resize(object):
    def __init__(self, target_size=None, min_side_length=None, max_side_length=None, max_side_bound=None):
        self.target_size = target_size
        self.min_side_length = min_side_length
        self.max_side_length = max_side_length
        self.max_side_bound = max_side_bound

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, Image.Image)
        assert labelmap is None or isinstance(labelmap, Image.Image)
        assert maskmap is None or isinstance(maskmap, Image.Image)

        width, height = img.size
        if self.target_size is not None:
            target_size = self.target_size
            w_scale_ratio = self.target_size[0] / width
            h_scale_ratio = self.target_size[1] / height

        elif self.min_side_length is not None:
            scale_ratio = self.min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        else:
            scale_ratio = self.max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        if self.max_side_bound is not None and  max(target_size) > self.max_side_bound:
            d_ratio = self.max_side_bound / max(target_size)
            w_scale_ratio = d_ratio * w_scale_ratio
            h_scale_ratio = d_ratio * h_scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] *= w_scale_ratio
            kpts[:, :, 1] *= h_scale_ratio

        if bboxes is not None and bboxes.size > 0:
            bboxes[:, 0::2] *= w_scale_ratio
            bboxes[:, 1::2] *= h_scale_ratio

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] *= w_scale_ratio
                    polygons[object_id][polygon_id][1::2] *= h_scale_ratio

        img = img.resize(target_size, Image.BILINEAR)
        if labelmap is not None:
            labelmap = labelmap.resize(target_size, Image.NEAREST)

        if maskmap is not None:
            maskmap = maskmap.resize(target_size, Image.NEAREST)

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class PILAugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> PILAugCompose([
        >>>     RandomCrop(),
        >>> ])
    """

    def __init__(self, configer, split='train'):
        self.configer = configer
        self.split = split

        self.transforms = dict()
        if self.split == 'train':
            shuffle_train_trans = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    train_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    for train_trans_seq in train_trans_seq_list:
                        shuffle_train_trans += train_trans_seq

                else:
                    shuffle_train_trans = self.configer.get('train_trans', 'shuffle_trans_seq')

            if 'random_saturation' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_saturation'] = RandomSaturation(
                    lower=self.configer.get('train_trans', 'random_saturation')['lower'],
                    upper=self.configer.get('train_trans', 'random_saturation')['upper'],
                    saturation_ratio=self.configer.get('train_trans', 'random_saturation')['ratio']
                )

            if 'random_hue' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hue'] = RandomHue(
                    delta=self.configer.get('train_trans', 'random_hue')['delta'],
                    hue_ratio=self.configer.get('train_trans', 'random_hue')['ratio']
                )

            if 'random_perm' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_perm'] = RandomPerm(
                    perm_ratio=self.configer.get('train_trans', 'random_perm')['ratio']
                )

            if 'random_contrast' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_contrast'] = RandomContrast(
                    lower=self.configer.get('train_trans', 'random_contrast')['lower'],
                    upper=self.configer.get('train_trans', 'random_contrast')['upper'],
                    contrast_ratio=self.configer.get('train_trans', 'random_contrast')['ratio']
                )

            if 'random_pad' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_pad'] = RandomPad(
                    up_scale_range=self.configer.get('train_trans', 'random_pad')['up_scale_range'],
                    pad_ratio=self.configer.get('train_trans', 'random_pad')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'padding' in self.configer.get('train_trans', 'trans_seq'):
                self.transforms['padding'] = Padding(
                    pad=self.configer.get('train_trans', 'padding')['pad'],
                    pad_ratio=self.configer.get('train_trans', 'padding')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value'),
                    allow_outside_center=self.configer.get('train_trans', 'padding')['allow_outside_center']
                )

            if 'random_brightness' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_brightness'] = RandomBrightness(
                    shift_value=self.configer.get('train_trans', 'random_brightness')['shift_value'],
                    brightness_ratio=self.configer.get('train_trans', 'random_brightness')['ratio']
                )

            if 'random_hsv' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hsv'] = RandomHSV(
                    h_range=self.configer.get('train_trans', 'random_hsv')['h_range'],
                    s_range=self.configer.get('train_trans', 'random_hsv')['s_range'],
                    v_range=self.configer.get('train_trans', 'random_hsv')['v_range'],
                    hsv_ratio=self.configer.get('train_trans', 'random_hsv')['ratio']
                )

            if 'random_gauss_blur' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_gauss_blur'] = RandomGaussBlur(
                    max_blur=self.configer.get('train_trans', 'random_gauss_blur')['max_blur'],
                    blur_ratio=self.configer.get('train_trans', 'random_gauss_blur')['ratio']
                )

            if 'random_hflip'  in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hflip'] = RandomHFlip(
                    swap_pair=self.configer.get('train_trans', 'random_hflip')['swap_pair'],
                    flip_ratio=self.configer.get('train_trans', 'random_hflip')['ratio']
                )

            if 'random_resize' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if self.configer.get('train_trans', 'random_resize')['method'] == 'random':
                    if 'max_side_bound' in self.configer.get('train_trans', 'random_resize'):
                        self.transforms['random_resize'] = RandomResize(
                            method=self.configer.get('train_trans', 'random_resize')['method'],
                            scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                            aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                            max_side_bound=self.configer.get('train_trans', 'random_resize')['max_side_bound'],
                            resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                        )
                    else:
                        self.transforms['random_resize'] = RandomResize(
                            method=self.configer.get('train_trans', 'random_resize')['method'],
                            scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                            aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                            resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                        )

                elif self.configer.get('train_trans', 'random_resize')['method'] == 'focus':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('train_trans', 'random_resize')['method'],
                        scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        target_size=self.configer.get('train_trans', 'random_resize')['target_size'],
                        resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                    )

                elif self.configer.get('train_trans', 'random_resize')['method'] == 'bound':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('train_trans', 'random_resize')['method'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        resize_bound=self.configer.get('train_trans', 'random_resize')['resize_bound'],
                        resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                    )

                else:
                    Log.error('Not Support Resize Method!')
                    exit(1)

            if 'random_crop' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if self.configer.get('train_trans', 'random_crop')['method'] == 'random':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('train_trans', 'random_crop')['method'] == 'center':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('train_trans', 'random_crop')['method'] == 'grid':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        grid=self.configer.get('train_trans', 'random_crop')['grid'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                else:
                    Log.error('Not Support Crop Method!')
                    exit(1)

            if 'random_rotate' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_rotate'] = RandomRotate(
                    max_degree=self.configer.get('train_trans', 'random_rotate')['rotate_degree'],
                    rotate_ratio=self.configer.get('train_trans', 'random_rotate')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'resize' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if 'target_size' in self.configer.get('train_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        target_size=self.configer.get('train_trans', 'resize')['target_size']
                    )
                if 'min_side_length' in self.configer.get('train_trans', 'resize'):
                    if 'max_side_bound' in self.configer.get('train_trans', 'resize'):
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('train_trans', 'resize')['min_side_length'],
                            max_side_bound=self.configer.get('train_trans', 'resize')['max_side_bound'],
                        )
                    else:
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('train_trans', 'resize')['min_side_length']
                        )
                if 'max_side_length' in self.configer.get('train_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        max_side_length=self.configer.get('train_trans', 'resize')['max_side_length']
                    )

        else:
            if 'random_saturation' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_saturation'] = RandomSaturation(
                    lower=self.configer.get('val_trans', 'random_saturation')['lower'],
                    upper=self.configer.get('val_trans', 'random_saturation')['upper'],
                    saturation_ratio=self.configer.get('val_trans', 'random_saturation')['ratio']
                )

            if 'random_hue' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hue'] = RandomHue(
                    delta=self.configer.get('val_trans', 'random_hue')['delta'],
                    hue_ratio=self.configer.get('val_trans', 'random_hue')['ratio']
                )

            if 'random_perm' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_perm'] = RandomPerm(
                    perm_ratio=self.configer.get('val_trans', 'random_perm')['ratio']
                )

            if 'random_contrast' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_contrast'] = RandomContrast(
                    lower=self.configer.get('val_trans', 'random_contrast')['lower'],
                    upper=self.configer.get('val_trans', 'random_contrast')['upper'],
                    contrast_ratio=self.configer.get('val_trans', 'random_contrast')['ratio']
                )

            if 'random_pad' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_pad'] = RandomPad(
                    up_scale_range=self.configer.get('val_trans', 'random_pad')['up_scale_range'],
                    pad_ratio=self.configer.get('val_trans', 'random_pad')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'padding' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['padding'] = Padding(
                    pad=self.configer.get('val_trans', 'padding')['pad'],
                    pad_ratio=self.configer.get('val_trans', 'padding')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value'),
                    allow_outside_center=self.configer.get('val_trans', 'padding')['allow_outside_center']
                )

            if 'random_brightness' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_brightness'] = RandomBrightness(
                    shift_value=self.configer.get('val_trans', 'random_brightness')['shift_value'],
                    brightness_ratio=self.configer.get('val_trans', 'random_brightness')['ratio']
                )

            if 'random_hsv' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hsv'] = RandomHSV(
                    h_range=self.configer.get('val_trans', 'random_hsv')['h_range'],
                    s_range=self.configer.get('val_trans', 'random_hsv')['s_range'],
                    v_range=self.configer.get('val_trans', 'random_hsv')['v_range'],
                    hsv_ratio=self.configer.get('val_trans', 'random_hsv')['ratio']
                )

            if 'random_gauss_blur' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_gauss_blur'] = RandomGaussBlur(
                    max_blur=self.configer.get('val_trans', 'random_gauss_blur')['max_blur'],
                    blur_ratio=self.configer.get('val_trans', 'random_gauss_blur')['ratio']
                )

            if 'random_hflip' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hflip'] = RandomHFlip(
                    swap_pair=self.configer.get('val_trans', 'random_hflip')['swap_pair'],
                    flip_ratio=self.configer.get('val_trans', 'random_hflip')['ratio']
                )

            if 'random_resize' in self.configer.get('val_trans', 'trans_seq'):
                if self.configer.get('val_trans', 'random_resize')['method'] == 'random':
                    if 'max_side_bound' in self.configer.get('val_trans', 'random_resize'):
                        self.transforms['random_resize'] = RandomResize(
                            method=self.configer.get('val_trans', 'random_resize')['method'],
                            scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                            aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                            max_side_bound=self.configer.get('val_trans', 'random_resize')['max_side_bound'],
                            resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                        )
                    else:
                        self.transforms['random_resize'] = RandomResize(
                            method=self.configer.get('val_trans', 'random_resize')['method'],
                            scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                            aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                            resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                        )

                elif self.configer.get('val_trans', 'random_resize')['method'] == 'focus':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('val_trans', 'random_resize')['method'],
                        scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        target_size=self.configer.get('val_trans', 'random_resize')['target_size'],
                        resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                    )

                elif self.configer.get('val_trans', 'random_resize')['method'] == 'bound':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('val_trans', 'random_resize')['method'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        resize_bound=self.configer.get('val_trans', 'random_resize')['resize_bound'],
                        resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                    )

                else:
                    Log.error('Not Support Resize Method!')
                    exit(1)

            if 'random_crop' in self.configer.get('val_trans', 'trans_seq'):
                if self.configer.get('val_trans', 'random_crop')['method'] == 'random':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('val_trans', 'random_crop')['method'] == 'center':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('val_trans', 'random_crop')['method'] == 'grid':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        grid=self.configer.get('val_trans', 'random_crop')['grid'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                else:
                    Log.error('Not Support Crop Method!')
                    exit(1)

            if 'random_rotate' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_rotate'] = RandomRotate(
                    max_degree=self.configer.get('val_trans', 'random_rotate')['rotate_degree'],
                    rotate_ratio=self.configer.get('val_trans', 'random_rotate')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'resize' in self.configer.get('val_trans', 'trans_seq'):
                if 'target_size' in self.configer.get('val_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        target_size=self.configer.get('val_trans', 'resize')['target_size']
                    )
                if 'min_side_length' in self.configer.get('val_trans', 'resize'):
                    if 'max_side_bound' in self.configer.get('val_trans', 'resize'):
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('val_trans', 'resize')['min_side_length'],
                            max_side_bound=self.configer.get('val_trans', 'resize')['max_side_bound'],
                        )
                    else:
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('val_trans', 'resize')['min_side_length']
                        )
                if 'max_side_length' in self.configer.get('val_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        max_side_length=self.configer.get('val_trans', 'resize')['max_side_length']
                    )

    def __check_none(self, key_list, value_list):
        for key, value in zip(key_list, value_list):
            if value == 'y' and key is None:
                return False

            if value == 'n' and key is not None:
                return False

        return True

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):

        if self.split == 'train':
            shuffle_trans_seq = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    shuffle_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    shuffle_trans_seq = shuffle_trans_seq_list[random.randint(0, len(shuffle_trans_seq_list))]
                else:
                    shuffle_trans_seq = self.configer.get('train_trans', 'shuffle_trans_seq')
                    random.shuffle(shuffle_trans_seq)

            for trans_key in (shuffle_trans_seq + self.configer.get('train_trans', 'trans_seq')):
                (img, labelmap, maskmap, kpts,
                 bboxes, labels, polygons) = self.transforms[trans_key](img, labelmap, maskmap,
                                                                        kpts, bboxes, labels, polygons)

        else:
            for trans_key in self.configer.get('val_trans', 'trans_seq'):
                (img, labelmap, maskmap, kpts,
                 bboxes, labels, polygons) = self.transforms[trans_key](img, labelmap, maskmap,
                                                                        kpts, bboxes, labels, polygons)

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'n', 'n', 'n', 'n', 'n']):
            return img

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['y', 'n', 'n', 'n', 'n', 'n']):
            return img, labelmap

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'n', 'n', 'y', 'n', 'n']):
            return img, bboxes

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'n', 'y', 'n', 'n', 'n']):
            return img, kpts

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'n', 'y', 'y', 'n', 'n']):
            return img, kpts, bboxes

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'y', 'y', 'n', 'n', 'n']):
            return img, maskmap, kpts

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['y', 'y', 'y', 'n', 'n', 'n']):
            return img, labelmap, maskmap, kpts

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'y', 'y', 'y', 'n', 'n']):
            return img, maskmap, kpts, bboxes

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['y', 'y', 'y', 'y', 'n', 'n']):
            return img, labelmap, maskmap, kpts, bboxes

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'n', 'n', 'y', 'y', 'n']):
            return img, bboxes, labels

        if self.__check_none([labelmap, maskmap, kpts, bboxes, labels, polygons], ['n', 'n', 'n', 'y', 'y', 'y']):
            return img, bboxes, labels, polygons

        Log.error('Params is not valid.')
        exit(1)
