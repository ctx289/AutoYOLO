import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.loaders import (LOADERS, LoadImages, LoadPilAndNumpy, LoadScreenshots, LoadStreams, LoadTensor,
                                      SourceTypes, autocast_list)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file

from .dataset import COCOYOLODataset
from ultralytics.data.utils import PIN_MEMORY
from torch.utils.data.dataset import ConcatDataset


def build_coco_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""
    return COCOYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)

