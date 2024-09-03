import itertools
import logging
import math
import operator
import os.path as osp
import re
import tempfile
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from pathlib import Path

import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from terminaltables import AsciiTable


@DATASETS.register_module()
class CocoDefectDataset(CocoDataset):
    CLASSES = None
    ATTRIBUTES = ["L1", "L2", "L3"]

    def __init__(self,
                 ann_file,
                 pipeline,
                 parse_ann_rules=None,
                 attributes=None,
                 attr_ignore_class=[],
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 filtered_poses=None,
                 crop_len_along_max=None,
                 crop_size=None,
                 crop_area=None,
                 crop_overlap=100,
                 need_fenji_classes=[],
                 ):
        """

        init function for CocoDefectDataset

        Args:
            parse_ann_rules (list, optional): list of rule cfgs. Defaults to None.
            Preliminary Knowledge:
            keys of annotation in coco json
                "image_id": 0,
                "id": 1,
                "bbox": [954, 729, 34, 82],
                "area": 2788,
                "segmentation": [[xxx, xxx, xxx, xxx, xxx]],
                "category_id": 2,
                "origin_label": "KL-QX-S4",
                "category_name": "KL",
                "hasSeg": false,
                "isOK": false,
                "isQX": true,
                "isMH": false,
                "isKBJ": false,
                "iscrowd": 0,
                "bbox_xyxy": [954, 729, 988, 811]

            Current Usage:
            Now we supporte two type of rules
            FilterByKeyArea
            dict(type='FilterByKeyArea', ignore_cfg)
                filter_cfg is a list with following format
                [
                    ['YS', ['isOK', 'isKBJ', 'area<2000']],
                    ['LW', ['isOK', 'isKBJ']],
                ]
            dict(type='ConvertCat', convert_cfg)
                convert_cfg is a dict, convert cat from key to value

            Example:
            parse_ann_rules=[
                dict(
                    type='IgnoreByKeyArea',
                    ignore_cfg={
                        'YW': ['isOK', 'isKBJ', 'area < 1600'],
                        'ZL': ['isOK', 'isKBJ', 'area < 1600'],
                        'YS': ['isOK', 'isKBJ', 'area < 1600'],
                        'QL': ['isOK', 'isKBJ', 'area < 1600'],
                        'HS': ['isOK', 'isKBJ', 'area < 1600'],
                        'LW': ['isOK', 'isKBJ'],
                        'KL': ['isOK', 'isKBJ'],
                        'QS': ['isOK', 'isKBJ'],
                        'BX': ['isOK', 'isKBJ'],
                        'DM': ['isKBJ', 'isOK & isL1', 'isOK & isL2'],
                    }),
                dict(
                    type='DeleteByKeyArea',
                    delete_cfg={
                        'ZL': ['isOK', 'isKBJ', 'area < 3000'],
                }),
                dict(
                    type='ClassToKBJ',
                    cls_to_kbj=['GS', 'FJ']
                ),
                dict(
                    type='ConvertCat',
                    convert_cfg={"KL":"LW", "YS":"LW"}
                ),
                dict(
                    type='ConvertCat',
                    convert_cfg={
                        10: 6,
                        11: 5,
                        12: 4,
                        13: 6,
                        14: 3
                    }
                ),
                dict(
                    type='ConvertIsokNg',
                    convert_isok_ng=['YW', 'ZL', 'YS']
                ),
            ]

            other args please visit CocoDataset

        How to crop:
            can only select one of [crop_size, crop_len_along_max, crop_area]
            below is listed by priority order
            crop_size is cropping a rectangle
            crop_len_along_max is cropping along longer side by pixels
            crop_area is cropping along longer side by area
        crop_size: crop size (width, height), such as (2000, 2000)
        crop_len_along_max: the max size of length to crop
        crop_area: same as crop_len_along_max, will be calculated as
            crop_len_along_max = crop_area / image_short_side
        ignore_cfg: make annotations ignore
        """
        if attributes is not None:
            if isinstance(attributes, tuple):
                attributes = list(attributes)
            assert isinstance(attributes, list)
            self.ATTRIBUTES = attributes
            self.need_fenji_classes = need_fenji_classes
        else:
            self.need_fenji_classes = []

        # add more params
        self.crop_len_along_max = crop_len_along_max
        self.crop_size = crop_size
        self.crop_area = crop_area
        self.crop_overlap = crop_overlap

        super(CocoDefectDataset, self).__init__(ann_file=ann_file,
                                                pipeline=pipeline,
                                                classes=classes,
                                                data_root=data_root,
                                                img_prefix=img_prefix,
                                                seg_prefix=seg_prefix,
                                                proposal_file=proposal_file,
                                                test_mode=test_mode,
                                                filter_empty_gt=filter_empty_gt)

        self._compare_dict = OrderedDict({
            '==': operator.eq,
            '<=': operator.le,
            ">=": operator.ge,
            '<': operator.lt,
            ">": operator.gt,
            '=': operator.eq,
        })

        self.parse_ann_rules = self._init_prepare_ann_parsing(parse_ann_rules)

        # attention that if ignore is [], self.coco.getCatIds will be full.
        self.attr_ignore_class = attr_ignore_class
        if self.attr_ignore_class == []:
            self.attr_ignore_class_id = []
        else:
            self.attr_ignore_class_id = self.coco.getCatIds(self.attr_ignore_class)

        # filter pose.
        if filtered_poses is not None:
            filtered_data_infos = []
            for data_info in self.data_infos:
                pose = int(re.search(r"P(\d+)", data_info['filename']).group(1))
                if pose in filtered_poses:
                    filtered_data_infos.append(data_info)
            self.data_infos = filtered_data_infos
            if not test_mode:
                self._set_group_flag()

    def _contain_compare_key(self, data):
        found = None
        for key in self._compare_dict.keys():
            if key in data:
                found = key
                break
        return found

    def _DeleteByKeyArea(self, delete_cfg, ann):
        rule_cfg = delete_cfg.get(ann['category_name'], None)
        if rule_cfg is None:
            return ann

        ann = deepcopy(ann)
        for single_rule in rule_cfg:
            # check if rule has comparison
            found_compare = self._contain_compare_key(single_rule)

            # if no comparison just check key is True or not
            if found_compare is None:
                if single_rule not in ann.keys():
                    raise ValueError(
                        f"[CocoDefectDataset] ann doesn't have key {single_rule}\
                            specified in parse_ann_rules"
                    )
                if ann[single_rule]:
                    return None
            # if has comparison, parse and compare
            else:
                compare_key, compare_value = single_rule.split(found_compare)
                compare_key = compare_key.strip(" ")
                compare_value = compare_value.strip(" ")
                try:
                    compare_value = eval(compare_value)
                except NameError:
                    pass
                if self._compare_dict[found_compare](ann[compare_key],
                                                     compare_value):
                    return None
        return ann

    def _IgnoreByKeyArea(self, ignore_cfg, ann):
        rule_cfg = ignore_cfg.get(ann['category_name'], None)
        if rule_cfg is None:
            return ann

        ann = deepcopy(ann)
        for single_rule in rule_cfg:
            # check if rule has comparison
            found_compare = self._contain_compare_key(single_rule)

            # if no comparison just check key is True or not
            if found_compare is None:
                # NOTE. modify by ryanwfu, logic to support &
                single_keys = single_rule.split('&')
                counts = 0
                for single_key in single_keys:
                    single_key = single_key.strip(" ")
                    if single_key not in ann.keys():
                        raise ValueError(
                            f"[CocoDefectDataset] ann doesn't have key {single_key}\
                                specified in parse_ann_rules"
                        )
                    if ann[single_key]:
                        counts += 1
                if counts == len(single_keys):
                    ann['iscrowd'] = 1
                    
                # if single_rule not in ann.keys():
                #     raise ValueError(
                #         f"[CocoDefectDataset] ann doesn't have key {single_rule}\
                #             specified in parse_ann_rules"
                #     )
                # if ann[single_rule]:
                #     ann['iscrowd'] = 1

            # if has comparison, parse and compare
            else:
                compare_key, compare_value = single_rule.split(found_compare)
                compare_key = compare_key.strip(" ")
                compare_value = compare_value.strip(" ")
                try:
                    compare_value = eval(compare_value)
                except NameError:
                    pass
                if self._compare_dict[found_compare](ann[compare_key],
                                                     compare_value):
                    ann['iscrowd'] = 1

        return ann

    def _ConvertCat(self, convert_cfg, ann):
        ann = deepcopy(ann)

        # try convert from name
        to_cat = convert_cfg.get(ann["category_name"], None)
        to_cat_id = None

        # try convert from id
        if to_cat is None:
            to_cat_id = convert_cfg.get(ann["category_id"], None)
        else:
            to_cat_id = self.cat_name_id[to_cat]

        if to_cat_id is not None:
            ann['category_id'] = to_cat_id
            ann["category_name"] = to_cat

        return ann

    def _ConvertIsokNg(self, convert_isok_ng, ann):
        ann = deepcopy(ann)
        src_class = ann['category_name']
        if src_class in convert_isok_ng:
            ann['isOK'] = False
        return ann

    def _ClassToKBJ(self, cls_to_kbj, ann):
        ann = deepcopy(ann)
        src_class = ann['category_name']
        if src_class in cls_to_kbj:
            ann['iscrowd'] = 1
            ann['isKBJ'] = True
        return ann

    def _init_prepare_ann_parsing(self, parse_ann_rules):
        if parse_ann_rules is None:
            return []

        rule_dict = {
            "IgnoreByKeyArea": self._IgnoreByKeyArea,
            "ConvertCat": self._ConvertCat,
            "DeleteByKeyArea": self._DeleteByKeyArea,
            "ConvertIsokNg": self._ConvertIsokNg,
            "ClassToKBJ": self._ClassToKBJ,
        }

        if isinstance(parse_ann_rules, dict):
            parse_ann_rules = [parse_ann_rules]

        if not isinstance(parse_ann_rules, list):
            raise ValueError(
                f"[CocoDefectDataset] parse_ann_rules is {type(parse_ann_rules)}, but expected list or dict")

        self.cat_name_id = dict()
        for cat_id, cat_name in self.coco.cats.items():
            self.cat_name_id[cat_name['name']] = cat_id

        rule_function = list()
        for rule_cfg in parse_ann_rules:
            rule_type = rule_cfg.pop("type")
            if rule_type not in rule_dict.keys():
                raise NotImplementedError(f"[CocoDefectDataset] Unsupported rule type {rule_type}")
            rule_function.append(partial(rule_dict[rule_type], **rule_cfg))
        return rule_function

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _ann2attrid(self, ann):
        if "origin_label" not in ann:
            print("origin_label is not in annotations")
            return -1
        #elif ann["category_id"] in self.attr_ignore_class_id:
        #    return -1
        else:
            if ann["origin_label"].split("-")[0] in self.need_fenji_classes:
                attr = ann["origin_label"].split("-")[1]
            else:
                return -1
            return self.ATTRIBUTES.index(attr) if attr in self.ATTRIBUTES else -1

    def _is_crop_valid(self, info_crop, ann_info, ng_only_crop_ng=True):
        ############# crop logic #########3
        # if drop other parts of ng images, drop here
        if ng_only_crop_ng and len(info_crop['image_cats']):
            has_intersect = False
            crop_range = info_crop['crop_range']
            for ann in ann_info:
                xmin, ymin, w, h = ann['bbox']
                xmax = xmin + w
                ymax = ymin + h

                # crop along longer side
                if len(crop_range) == 2:
                    if info_crop['height'] > info_crop['width']:
                        if (ymax > crop_range[0] and ymax < crop_range[1]) or \
                            (ymin > crop_range[0] and ymin < crop_range[1]) or \
                            (ymin < crop_range[0] and ymax > crop_range[1]) or \
                            (ymin > crop_range[0] and ymax < crop_range[1]):
                            has_intersect = True
                            break
                    else:
                        if (xmax > crop_range[0] and xmax < crop_range[1]) or \
                            (xmin > crop_range[0] and xmin < crop_range[1]) or \
                            (xmin < crop_range[0] and xmax > crop_range[1]) or \
                            (xmin > crop_range[0] and xmax < crop_range[1]):
                            has_intersect = True
                            break
                # crop by rectangle
                else:
                    if ((ymax > crop_range[1] and ymax < crop_range[3]) or \
                            (ymin > crop_range[1] and ymin < crop_range[3]) or \
                            (ymin < crop_range[1] and ymax > crop_range[3]) or \
                            (ymin > crop_range[1] and ymax < crop_range[3])) and \
                            ((xmax > crop_range[0] and xmax < crop_range[2]) or \
                            (xmin > crop_range[0] and xmin < crop_range[2]) or \
                            (xmin < crop_range[0] and xmax > crop_range[2]) or \
                            (xmin > crop_range[0] and xmax < crop_range[2])):
                        has_intersect = True
                        break
            return has_intersect
        return True

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            ann_info = self.coco.load_anns(ann_ids)

            # if no crop
            if self.crop_len_along_max is None and self.crop_area is None and self.crop_size is None:
                data_infos.append(info)

            # if crop rectangle
            elif self.crop_size is not None:
                width_crop_num = info['width'] // self.crop_size[0]
                height_crop_num = info['height'] // self.crop_size[0]
                for crop_widx in range(width_crop_num):
                    for crop_hidx in range(height_crop_num):
                        info_crop = deepcopy(info)
                        info_crop['crop_range'] = (
                            max(0, crop_widx * self.crop_size[0] - self.crop_overlap),
                            max(0, crop_hidx * self.crop_size[1] - self.crop_overlap),
                            min(info['width'], (crop_widx + 1) * self.crop_size[0] + self.crop_overlap),
                            min(info['height'], (crop_hidx + 1) * self.crop_size[1] + self.crop_overlap)
                        )
                        if self._is_crop_valid(info_crop, ann_info):
                            data_infos.append(info_crop)

            # if crop along max
            else:
                min_image_side = min([info['height'], info['width']])
                max_image_side = max([info['height'], info['width']])
                if self.crop_len_along_max is None:
                    crop_len_along_max = math.ceil(self.crop_area // min_image_side)
                else:
                    crop_len_along_max = self.crop_len_along_max
                # if has crop
                crop_num = math.ceil(max_image_side // crop_len_along_max)
                for crop_idx in range(crop_num):
                    info_crop = deepcopy(info)
                    info_crop['crop_range'] = (
                        max(0, crop_idx * crop_len_along_max -
                            self.crop_overlap),
                        min(max_image_side, (crop_idx + 1) * crop_len_along_max +
                            self.crop_overlap))
                    if self._is_crop_valid(info_crop, ann_info):
                        data_infos.append(info_crop)

            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _preprocess_ann(self, ann):
        # change ann
        if self.parse_ann_rules is None:
            return ann

        for rule_func in self.parse_ann_rules:
            ann = rule_func(ann=ann)
            if ann is None:
                break
        return ann

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_attrs = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            ann = self._preprocess_ann(ann)
            if ann is None:
                continue

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_attrs.append(self._ann2attrid(ann))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            attrs=gt_attrs)

        return ann

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                # has attrs
                if isinstance(result[label], list):
                    bboxes = result[label][0]
                    attrs = result[label][1]
                else:
                    bboxes = result[label]
                    attrs = None

                for i in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(bboxes[i][4])
                    data["category_id"] = self.cat_ids[label]

                    if attrs is not None:
                        if len(attrs[i]) > len(self.ATTRIBUTES):
                            data["attribute_id"] = np.argmax(attrs[i][:-1])
                        else:
                            data["attribute_id"] = np.argmax(attrs[i])
                        data["attribute_score"] = attrs[i][data["attribute_id"]]
                        data["attribute_name"] = self.ATTRIBUTES[data["attribute_id"]]

                    json_results.append(data)
        return json_results
