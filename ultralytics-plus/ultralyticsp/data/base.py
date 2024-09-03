import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import re
import numpy as np
import psutil
from torch.utils.data import Dataset

from tqdm import tqdm
import random

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT
from ultralytics.data.utils import HELP_URL, IMG_FORMATS
from ultralytics.data import BaseDataset

import json
from collections import defaultdict


class COCOBaseDataset(BaseDataset, Dataset):

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0,
                 data_info=None):
        Dataset.__init__(self)
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction

        # NOTE. 2023/08/17 modified by ryanwfu
        self.is_train = augment
        if data_info is not None and data_info['coco_format_json']:
            self.coco_format_json = data_info['coco_format_json']
            self.roi_crop = data_info['roi_crop']
            self.allow_empty = data_info['allow_empty']
            self.roi_key = data_info['roi_key']
            self.roi_offset = data_info.get('roi_offset', None)
            self.im_files, self.labels = self.parser_coco_format_json(self.img_path, data_info)
            LOGGER.info(f'DATA INFO:\t allow_empty: {self.allow_empty if self.is_train else True}\t'
                    f'roi_crop: {self.roi_crop}\t roi_key: {self.roi_key}\t roi_offset: {self.roi_offset if self.is_train else None}\t')
        else:
            self.im_files = self.get_img_files(self.img_path)
            self.labels = self.get_labels()
        
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache stuff
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)
    
    def parser_coco_format_json(self, img_path, data_info):
        im_files = []
        labels = []
        if img_path == data_info['train']:
            img_prefix = data_info['train_prefix']
        elif img_path == data_info['val']:
            img_prefix = data_info['val_prefix']
        elif img_path == data_info['test']:
            img_prefix = data_info['test_prefix']
        else:
            raise RuntimeError("Something wrong, the dataset path {img_path} does not match to data info get from yaml.")
        
        if isinstance(img_path, str):
            img_path = [img_path]
        assert isinstance(img_path, list)
        
        for coco_json_path in img_path:

            with Path(coco_json_path).open('r') as f:
                coco_data = json.load(f)

            # category to cat id mapping
            categories = coco_data['categories']
            catid2catname = {cat_info['id']: cat_info['name'] for cat_info in categories}

            # NOTE. Map id -> catid according to the category field in the coco json file, deprecated.
            # ori_id2catid_mapping = {}
            # for i, cat_info in enumerate(categories):
            #     ori_id2catid_mapping[cat_info['id']] = i

            # category name to cat id mapping
            catname2catid_mapping = {}
            for cat_index, cat_name in data_info['names'].items():
                catname2catid_mapping[cat_name] = cat_index

            # Create image dict
            images = {'%g' % x['id']: x for x in coco_data['images']}
            # Create image-annotations dict
            imgToAnns = defaultdict(list)
            for image_id in images.keys():
                imgToAnns[int(image_id)] = []
            for ann in coco_data['annotations']:
                imgToAnns[ann['image_id']].append(ann)

            # Write labels file
            for img_id, anns in tqdm(imgToAnns.items()):
                img = images['%g' % img_id]
                h, w, f = img['height'], img['width'], img['file_name']

                im_file_ = os.path.join(img_prefix, f)
                img_info_ = {'im_file': im_file_,
                            'shape': (h, w),    # shape is for rect, this shape may be poped by self.set_rectangle.
                            'shape2': (h, w),   # NOTE. shape2 is for check, 2023/09/05 added by ryanwfu
                            'cls': None, 
                            'bboxes': None, 
                            'segments': [], 
                            'keypoints': None, 
                            'normalized': True, 
                            'bbox_format': 'xywh',
                            'roi_info': None,
                            'image_id': img_id,}
                
                if self.roi_crop:
                    if self.roi_key in img and img[self.roi_key]:
                        x1, y1, x2, y2 = img[self.roi_key]
                        
                        # NOTE. modified by ryanwfu. 2023/09/21
                        if self.is_train and self.roi_offset:
                            crop_w, crop_h = x2 - x1, y2 - y1
                            offset_x = int(crop_w * self.roi_offset)
                            offset_y = int(crop_h * self.roi_offset)
                            if x2 - offset_x > x1 + offset_x:
                                x1 = max(random.randint(x1 - offset_x, x1 + offset_x), 0)
                                x2 = min(random.randint(x2 - offset_x, x2 + offset_x), w - 1)
                            if y2 - offset_y > y1 + offset_y:
                                y1 = max(random.randint(y1 - offset_y, y1 + offset_y), 0)
                                y2 = min(random.randint(y2 - offset_y, y2 + offset_y), h - 1)
                        
                        img_info_['roi_info'] = [x1, y1, x2, y2]
                    else:
                        img_info_['roi_info'] = [0, 0, w, h]

                bboxes = []
                for ann in anns:
                    if ann['iscrowd']:
                        continue

                    ## ignore some items
                    if 'ignore' in data_info:
                        ignore_flag = False
                        ignore_items = data_info['ignore']
                        for ignore_item in ignore_items:
                            if ann.get(ignore_item):
                                ignore_flag = True
                                break
                        if ignore_flag:
                            continue

                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(ann['bbox'], dtype=np.float64)
                    if self.roi_crop:
                        roi = img_info_['roi_info']

                        # NOTE. modified by ryanwfu 2023/09/25, crop bboxes accordingly and clip to the boundary
                        box_xyxy = box + np.array([0, 0, box[0], box[1]], dtype=np.float64)
                        temp_x1 = max(box_xyxy[0], roi[0])
                        temp_y1 = max(box_xyxy[1], roi[1])
                        temp_x2 = min(box_xyxy[2], roi[2])
                        temp_y2 = min(box_xyxy[3], roi[3])
                        if temp_x2 <= temp_x1 or temp_y2 <= temp_y1:
                            continue
                        box_xyxy = np.array([temp_x1, temp_y1, temp_x2, temp_y2], dtype=np.float64)
                        box = box_xyxy - np.array([0, 0, box_xyxy[0], box_xyxy[1]], dtype=np.float64)

                        box[:2] += box[2:] / 2   # xy top-left corner to center
                        box[:2] -= np.array(roi[:2], dtype=np.float64) 
                        croped_w, croped_h = np.array(roi[2:], dtype=np.float64) - np.array(roi[:2], dtype=np.float64)
                        box[[0, 2]] /= croped_w  # normalize x
                        box[[1, 3]] /= croped_h  # normalize y

                        # NOTE. modified by ryanwfu 2023/09/19, support rect from now on.
                        img_info_['shape'] = (croped_h, croped_w)
                    else:
                        box[:2] += box[2:] / 2  # xy top-left corner to center
                        box[[0, 2]] /= w  # normalize x
                        box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue

                    cat_name_ = catid2catname[ann['category_id']]
                    if catname2catid_mapping.get(cat_name_) is not None:
                        cls = catname2catid_mapping[cat_name_]
                    else:
                        continue

                    box = [cls] + box.tolist()
                    # NOTE. modified by ryanwfu 2023/09/25
                    bboxes.append(box)

                temp_info_ = np.array(bboxes, dtype=np.float32)
                try:
                    img_info_['cls'] = temp_info_[:, 0][:, np.newaxis]
                    img_info_['bboxes'] = temp_info_[:, 1:]
                except IndexError as e:
                    assert len(temp_info_) == 0
                    temp_info_ = np.zeros((0, 5), dtype=np.float32)
                    img_info_['cls'] = temp_info_[:, 0][:, np.newaxis]
                    img_info_['bboxes'] = temp_info_[:, 1:]
                
                if self.is_train and not self.allow_empty and len(img_info_['cls']) == 0:
                    continue

                im_files.append(im_file_)
                labels.append(img_info_)
        return im_files, labels

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image

                # NOTE. modified by ryanwfu. 2023/09/28, compatible with Chinese paths in windows
                fb = f.encode('utf-8')
                im = cv2.imdecode(np.fromfile(fb, dtype=np.uint8), cv2.IMREAD_COLOR)
                # im = cv2.imread(f)  # BGR

                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw

            # Data shape check. NOTE. 2023/09/05 added by ryanwfu;
            # NOTE. modified by ryanwfu 2023/09/25. 1. Replace with warning; 2. Put data shape check to converter_to_coco; 
            if (h0, w0) != self.labels[i]['shape2']:
                LOGGER.warning(f"WARNING ⚠️ 'The real shape {(h0, w0)} of image data does not match the height and weight {self.labels[i]['shape2']} in coco json'")
            
            # modified by ryanwfu
            if self.roi_crop:
                label_ = self.labels[i]
                roi_ = label_['roi_info']
                im = im[roi_[1]:roi_[3], roi_[0]:roi_[2]]
                h0, w0 = im.shape[:2]

            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                                interpolation=interp)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]              # im, hw_original, hw_resized

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
    
    # NOTE. modified by ryanwfu. 2023/09/28, compatible with Chinese paths in windows
    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            # NOTE. modified by ryanwfu. 2023/09/28, replace cv2.imread with cv2.imdecode
            np.save(f.as_posix(), cv2.imdecode(np.fromfile(self.im_files[i].encode('utf-8'), dtype=np.uint8), cv2.IMREAD_COLOR))

    # NOTE. modified by ryanwfu. 2023/09/28, compatible with Chinese paths in windows
    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            # NOTE. modified by ryanwfu. 2023/09/28, replace cv2.imread with cv2.imdecode
            im = cv2.imdecode(np.fromfile(random.choice(self.im_files).encode('utf-8'), dtype=np.uint8), cv2.IMREAD_COLOR)

            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache
