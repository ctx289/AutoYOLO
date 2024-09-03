import math
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa

from ultralytics.utils import LOGGER
from ultralytics.data.augment import (Compose, Mosaic, CopyPaste, RandomPerspective, LetterBox,
                                      MixUp, Albumentations, RandomHSV, RandomFlip)


class CopyPasteDet:

    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def bboxes_to_segments(self, bboxes):
        polygons = []
        for bbox in bboxes:
            polygon = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            polygons.append(polygon)
        return np.array(polygons)
    
    def random_ins(self, instances, h, w):
        assert instances._bboxes.format == 'xyxy'
        for i in range(len(instances.bboxes)):
            x1, y1, x2, y2 = instances.bboxes[i]
            b_w, b_h = x2 - x1, y2 - y1
            _x1 = random.randint(1, int(w - b_w - 1))
            _y1 = random.randint(1, int(h - b_h - 1))
            _x2 = _x1 + b_w
            _y2 = _y1 + b_h
            instances._bboxes.bboxes[i] = np.array([_x1, _y1, _x2, _y2])
        return instances

    def __call__(self, labels):
        """Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)."""
        im = labels['img']
        cls = labels['cls']
        h, w = im.shape[:2]
        instances = labels.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        if self.p and len(instances.bboxes):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = im.copy()

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip = self.random_ins(ins_flip, h, w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.50).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)

                for i, bbox in enumerate(instances.bboxes[[j]]):
                    bbox_img = im_new[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()
                    target_bbox = ins_flip.bboxes[[j]][i].astype(int)
                    im[target_bbox[1]:target_bbox[1]+bbox_img.shape[0], target_bbox[0]:target_bbox[0]+bbox_img.shape[1]] = bbox_img

        labels['img'] = im
        labels['cls'] = cls
        labels['instances'] = instances
        return labels
    

def custom_v8_transforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste),
        CopyPasteDet(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = dataset.data.get('flip_idx', [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms
