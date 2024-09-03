# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mmcv
import numpy as np
from mmcv.utils import Registry, print_log
from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from terminaltables import AsciiTable
from .coco_attr_dataset import AttrCocoDataset
from .builder import ATTR_DATASET


@ATTR_DATASET.register_module()
class IndSAttrDataset(AttrCocoDataset):
    CLASSES = None
    ATTRIBUTES = ["S1", "S2", "S3", "S4"]

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        attributes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        attr_ignore_class=None,
    ):
        super().__init__(
            ann_file,
            pipeline,
            classes,
            attributes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
        )
        self.attr_ignore_class = attr_ignore_class
        self.attr_ignore_class_id = self.coco.getCatIds(self.attr_ignore_class)

    def ann2attrid(self, ann):
        if "origin_label" not in ann:
            print("origin_label is not in annotations")
            return -1
        elif ann["category_id"] in self.attr_ignore_class_id:
            return -1
        else:
            attr = ann["origin_label"].split("-")[-1]
            return self.ATTRIBUTES.index(attr) if attr in self.ATTRIBUTES else -1
