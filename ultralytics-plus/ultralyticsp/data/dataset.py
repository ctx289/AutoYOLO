# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import json
from collections import defaultdict

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable
from ultralytics.data.augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label

from ultralytics.data import YOLODataset
from .base import COCOBaseDataset

from ultralyticsp.data.augment import custom_v8_transforms


class COCOYOLODataset(YOLODataset, COCOBaseDataset):
    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        # brorrow from YOLODataset
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        COCOBaseDataset.__init__(self, *args, **kwargs, data_info=data)
    
    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # NOTE. modified by ryanwfu 2023/10/11. Support CopyPasteDet.
            transforms = custom_v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms
