import logging
import math
import os
from functools import partial

import cv2
import numpy as np
import yaml

from ..builder import MODULES
from ..utils.error_code import error_code_dict
from .base_detector import BaseDetector
from ..builder import build_module


@MODULES.register_module()
class PPFasterDetector(BaseDetector):

    def __init__(
            self,
            config,
            ckpt,
            classes,
            keep_cats=None,
            poses=None,
            roi_dir=None,
            roi_cfg=dict(type="LoadROISingle"),
            # nodes
            input_name='images',
            output_name='preds',
            # postprocess
            min_wh=1,
            crop_by_seg=False,
            crop_by_seg_offset=0,
            max_per_img=50,
            min_thresh=0.1,
            # others
            gpu_id=0,
            verbose=False):
        """ Init for mmdetection detection class
            other args please visit base_detector.py
        Args:
            optimize (bool, optional): whether use tiacc optimize. Defaults to False.
            use_tiacc (bool, optional): the same as optimize, but compatible
                with pingtai.
            optimize_input_shape (list, optional): input shape.
                Defaults to [{ 'seperate': '1*3*1280*1536' }].
            encrypt (dict, optional): whether encrypt
            encrypt_platform (bool, optional): whether decrpyt as platform format.
                Defaults to False.

            rotate (str, optional): mode to rotate, has two mode.
                must in [h>w, w>h]
                h>w: rotate to make sure iamge h > w
                w>h: rotate to make sure image w > h
                Defaults to None.
            crop (dict, optional): how to crop data
                if pose specified will crop othterwise will not
                dict(
                    1: 3, # if pose 1, crop to 3 images along long side
                    2: 2, # if pose 2, crop to 2 images along long side
                    3: 1  # if pose 3, crop to 1 images along long side
                )
            crop_overlap (int, optional): crop_overlap of crops

            min_wh (int, optional): min length of detected bbox side. default to 1

        """
        self.config = config
        self.ckpt = ckpt
        self.classes = classes
        self.input_name = input_name
        self.output_name = output_name
        self.keep_cats = keep_cats
        self.poses = poses
        self.verbose = verbose
        self.gpu_id = gpu_id

        self.roi_cfg = roi_cfg
        self.roi_dir = roi_dir
        if self.roi_dir is not None:
            roi_cfg['roi_dir'] = roi_dir
            self.roi_filter = build_module(roi_cfg)

        self.detector = self.init(
            self.config, self.ckpt, device_id=self.gpu_id)
        
        self.min_wh = min_wh
        self.crop_by_seg = crop_by_seg
        self.crop_by_seg_offset = crop_by_seg_offset
        self.min_thresh = min_thresh
        self.max_per_img = max_per_img

    def init(self, config, ckpt, device_id):
        import paddle
        from pp_deploy_python import Detector
        paddle.enable_static()
        with open(config) as f:
            yml_conf = yaml.safe_load(f)
            arch = yml_conf['arch']
            detector_func = 'Detector'
        
        model_dir = ckpt
        detector = eval(detector_func)(
            model_dir,
            device='GPU',
            # NOTE. Modify by ryanwfu
            device_id=device_id,
            run_mode='paddle',
            batch_size=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            cpu_threads=1,
            enable_mkldnn=False,
            enable_mkldnn_bfloat16=False,
            threshold=0.1,
            output_dir=None,
            use_fd_format=False)

        return detector
    
    def _set_device(self, device_type: str, device_id: int = 0) -> None:
        """
        for sdk _set_device
        """
        print(f'set PPFasterDetector device to {device_type}:{device_id}')
        del self.detector
        self.detector = self.init(
            self.config, self.ckpt, device_id=device_id)

    def _inference(self, image, crop_infer=False, offset=None):
        """ get mmdet inference results with format

        Args:
            image (np.array)
        """
        # from IPython import embed;embed()
        if crop_infer:
            x1, x2, y1, y2 = offset
            results = self.detector.predict_numpy([image[y1:y2, x1:x2, :]])
        else:
            results = self.detector.predict_numpy([image])

        # refers to ppdet/metrics/json_results.py -> get_det_res
        boxes, boxes_num = results['boxes'], results['boxes_num'][0]
        if self.max_per_img < boxes_num:
            boxes_num = self.max_per_img
            boxes = boxes[:boxes_num,:]
        pred_list = []
        for j in range(boxes_num):
            dt = boxes[j]
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0 or score < self.min_thresh:
                continue
            label = int(num_id)
            if (xmin >= xmax - self.min_wh) or (ymin >= ymax - self.min_wh):
                continue
            area = (ymax - ymin) * (xmax - xmin)
            length=np.sqrt((ymax - ymin) ** 2 + (xmax - xmin) ** 2)
            if crop_infer:
                xmin += x1
                ymin += y1
                xmax += x1
                ymax += y1
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            boundary = None

            # keep cats if needed
            if self.keep_cats is not None and self.classes[
                    label] not in self.keep_cats:
                continue

            pred_list.append({
                "det_score": float(score),
                "det_bbox": bbox,
                "det_code": self.classes[label],
                "area": int(area),
                "length": int(length),
                "polygon": boundary,
            })

        return pred_list

    def predict(self, feed_dict, **kwargs):

        images = feed_dict[self.input_name]
        if isinstance(images, (list, tuple)):
            image = images[0]
        else:
            image = images

        crop_infer = False
        if self.crop_by_seg and "roi_seg" in feed_dict.keys() and feed_dict['roi_seg']['roi_seg_ok']:
            if 'roi_polygon' in feed_dict['roi_seg'] and feed_dict['roi_seg']['roi_polygon'] is not None:
                offset = self.crop_by_seg_offset
                polygon = feed_dict['roi_seg']['roi_polygon']
                all_xy = sum(polygon,[])
                if all_xy:
                    all_x = all_xy[::2]
                    all_y = all_xy[1::2]
                    min_xy = (min(all_x), min(all_y))
                    max_xy = (max(all_x), max(all_y))
                    x1 = max(min_xy[0]-offset,0)
                    x2 = min(max_xy[0]+offset,image.shape[1]-1)
                    y1 = max(min_xy[1]-offset,0)
                    y2 = min(max_xy[1]+offset,image.shape[0]-1)
                    crop_infer = True

        if crop_infer:
            pred_list = self._inference(image, crop_infer, [x1, x2, y1, y2])
        else:
            pred_list = self._inference(image)

        self.feed_data(feed_dict, self.output_name, pred_list)
        return feed_dict
