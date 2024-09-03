import logging
import math
import os
from functools import partial

import cv2
import glob
import numpy as np
import yaml
import json
from pathlib import Path

from ..builder import MODULES
from ..utils.error_code import error_code_dict
from .base_detector import BaseDetector
from ..builder import build_module
from ..utils.pose_utils import get_pose


@MODULES.register_module()
class UltralyticsFasterDetector(BaseDetector):

    def __init__(
            self,
            config,
            ckpt,
            classes,
            keep_cats=None,
            poses=None,

            # nodes
            input_name='images',
            output_name='preds',

            # optimize - deprecated here
            use_tiacc=False,
            optimize=False, # deprecated
            optimize_input_shape=[{
                'seperate': '1*3*1280*1536'
            }],

            # encrypt
            encrypt=False,
            encrypt_platform=False,

            # preprocess - deprecated here
            rotate=None,
            crop=None,
            crop_overlap=50,
            crop_area=None,

            # preprocess - crop_infer NOTE. 2023/08/30
            crop_by_seg=False,
            crop_by_outer=False,
            roi_dir=None,
            crop_offset=0,
            
            # deprecated
            roi_info={},
            roi_cfg=dict(type="LoadROISingle"),
            
            # postprocess
            min_wh=1,
            min_thresh=0.01,
            max_det=100,
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

            if crop_by_outer and crop_by_seg be both True, crop_by_seg takes precedence

        """
        self.encrypt = encrypt
        self.encrypt_platform = encrypt_platform
        self.min_wh = min_wh
        self.min_thresh = min_thresh
        self.max_det = max_det
        super(UltralyticsFasterDetector, self).__init__(config=config,
                                               ckpt=ckpt,
                                               classes=classes,
                                               input_name=input_name,
                                               output_name=output_name,
                                               keep_cats=keep_cats,
                                               poses=poses,
                                               roi_dir=roi_dir,
                                               roi_cfg=roi_cfg,
                                               gpu_id=gpu_id,
                                               verbose=verbose)
        
        self.crop_by_seg = crop_by_seg
        self.crop_offset = crop_offset
        self.crop_by_outer = crop_by_outer
        self.outers = self.init_outers(
            self.crop_by_outer, self.roi_dir)
        
        # NOTE. modified by ryanwfu 2023/09/15
        # Custom inference is a simplified ultralytics inference method. 
        # When the size of the image is large, compared with the original
        # method, the speed is significantly improved.
        self.custom_inference = True

        # NOTE. modified by ryanwfu 2023/09/26
        # When classes and model.names are inconsistent, 
        # the results will filter out categories that are not in classes
        # put this in here or to FilterByKeyAndScore - wlist_ok
        self.names = self.model.names
        if self.classes is None:
            self.classes = list(self.names.values())

    def init(self, config, ckpt, device):
        """
        config is no use for Ultralytics;
        device is not used here;
        """
        from ultralyticsp.models import CustomYOLO, CustomRTDETR
        from ultralyticsp.custom import yaml_load   
        
        if self.encrypt:
            config, ckpt = self._encrypt_bilevel(config, ckpt)
        elif self.encrypt_platform:
            config, ckpt = self._encrypt_platform(config, ckpt)

        if isinstance(config, (str, Path)):
            cfg = yaml_load(config)
        elif isinstance(config, dict):
            cfg = config
        else:
            raise TypeError('config must be a filename or dict object, '
                        f'but got {type(config)}')

        if 'yolo' in os.path.basename(cfg['model']):
            detector = CustomYOLO(ckpt)
        elif 'rtdetr' in os.path.basename(cfg['model']):
            detector = CustomRTDETR(ckpt)
        else:
            raise NotImplementedError('{} is not supproted'.format(os.path.basename(cfg['model'])))
        
        if self.verbose:
            logging.info("Initializing UltralyticsFasterDetector on {}".format(device)) 
            detector.info()

        # NOTE. warm_up and put the model into the specified GPU
        random_matrix = (np.random.random((640, 640, 3)) * 255).astype(np.uint8)
        _ = detector.predict([random_matrix], conf=self.min_thresh, verbose=False, device=self.gpu_id)
        
        return detector

    def init_outers(self, crop_by_outer, roi_dir):
        """
        crop_by_outer: True / False
        roi_info:{ # roi配置
            "roi_crop": true,   # 固定为true
            "roi_key": "outer"  # 固定
            "roi_data": {
                "P01": "/xxxx/P01.json"  # labelme json
            }
        """
        outers = dict()
        if crop_by_outer and roi_dir is not None:
            roi_key = 'outer'
            for json_path in glob.glob(os.path.join(roi_dir, "*.json")):
                pose = get_pose(os.path.basename(json_path))
                with open(json_path, 'r') as f:
                    content = json.load(f)
                for shape in content["shapes"]:
                    if shape['label'] == roi_key:
                        if shape['shape_type'] == 'rectangle':
                            points = np.array(shape['points'])
                            all_x = points[:, 0]
                            all_y = points[:, 1]
                            outers[pose] = [int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y))]
                        elif shape['shape_type'] == 'polygon':
                            points = np.array(shape['points'])
                            all_x = points[:, 0]
                            all_y = points[:, 1]
                            outers[pose] = [int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y))]
                        else:
                            raise Exception("Unsupported shape_type of {} in\
                                             {}".format(roi_key, json_path))
                        # find one then break
                        break
        return outers
    
    def _set_device(self, device_type: str, device_id: int = 0) -> None: 
        """
        for sdk _set_device
        """
        print(f'set UltralyticsFasterDetector device to {device_type}:{device_id}')
        self.gpu_id = device_id
        # NOTE. warm_up and put the model into the specified GPU
        random_matrix = (np.random.random((640, 640, 3)) * 255).astype(np.uint8)
        _ = self.model.predict([random_matrix], conf=self.min_thresh, verbose=False, device=self.gpu_id)

    def _inference(self, image, crop_infer=False, offset=None):
        """ get mmdet inference results with format

        Args:
            image (np.array)
        """
        if crop_infer:
            x1, y1, x2, y2 = offset
            results = self.model.predict([image[y1:y2, x1:x2, :]], conf=self.min_thresh, max_det=self.max_det, \
                                         verbose=False, custom_inference=self.custom_inference, device=self.gpu_id)[0]
        else:
            results = self.model.predict([image], conf=self.min_thresh, verbose=False, max_det=self.max_det, \
                                         custom_inference=self.custom_inference, device=self.gpu_id)[0]

        if self.custom_inference:
            results['boxes'] = results['boxes'].cpu()
            xyxys = results['boxes'][:, :4]
            confs = results['boxes'][:, -2]
            labels = results['boxes'][:, -1]
        else:
            results = results.cpu()
            xyxys = results.boxes.xyxy
            confs = results.boxes.conf
            labels = results.boxes.cls

        pred_list = []
        for i in range(len(labels)):
            xmin, ymin, xmax, ymax = xyxys[i]
            w, h = xmax - xmin, ymax - ymin
            conf = float(confs[i])
            label = int(labels[i])
            if label < 0:
                continue
            if (h <= self.min_wh) or (w <= self.min_wh):
                continue
            area = w * h
            length = np.sqrt((ymax - ymin) ** 2 + (xmax - xmin) ** 2)
            if crop_infer:
                xmin += x1
                ymin += y1
                xmax += x1
                ymax += y1
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            boundary = None

            # keep cats if needed
            if self.keep_cats is not None and self.names[
                    label] not in self.keep_cats:
                continue
            
            # NOTE. modified by ryanwfu 2023/09/26
            # put to FilterByKeyAndScore - wlist_ok
            # if self.names[label] not in self.classes:
            #     continue

            pred_list.append({
                "det_score": float(conf),
                "det_bbox": bbox,
                "det_code": self.names[label],
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
        if self.crop_by_seg and "roi_seg" in feed_dict.keys() and \
              feed_dict['roi_seg']['roi_seg_ok'] and feed_dict['roi_seg']['roi_polygon'] is not None:
            offset = self.crop_offset
            polygon = feed_dict['roi_seg']['roi_polygon']
            all_xy = sum(polygon,[])
            if all_xy:
                all_x = all_xy[::2]
                all_y = all_xy[1::2]
                min_xy = (min(all_x), min(all_y))
                max_xy = (max(all_x), max(all_y))
                x1 = max(min_xy[0] - offset, 0)
                x2 = min(max_xy[0] + offset, image.shape[1] - 1)
                y1 = max(min_xy[1] - offset, 0)
                y2 = min(max_xy[1] + offset, image.shape[0] - 1)
                crop_infer = True

        elif self.crop_by_outer and feed_dict['pose'] in self.outers:
            x1, y1, x2, y2 = self.outers[feed_dict['pose']]
            offset = self.crop_offset
            x1 = max(x1 - offset, 0)
            x2 = min(x2 + offset, image.shape[1] - 1)
            y1 = max(y1 - offset, 0)
            y2 = min(y2 + offset, image.shape[0] - 1)
            crop_infer = True

        if crop_infer:
            pred_list = self._inference(image, True, [x1, y1, x2, y2])
        else:
            pred_list = self._inference(image)

        detector_message={
            'crop_by_seg': self.crop_by_seg,
            'crop_by_outer': self.crop_by_outer,
            'crop_infer': crop_infer,
            'crop_outer': [x1, y1, x2, y2] if crop_infer else None,
        }
            
        feed_dict['detector_message'] = detector_message
        self.feed_data(feed_dict, self.output_name, pred_list)
        return feed_dict