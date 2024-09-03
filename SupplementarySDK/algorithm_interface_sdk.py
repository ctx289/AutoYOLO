#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import os
import yaml

# init sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/3rd_party')
print('Add $SDK_PATH/3rd_party to sys.path.')
# init IND_PROJ_PATH2
os.environ['IND_PROJ_PATH2'] = os.path.dirname(os.path.abspath(__file__))
print('Setting IND_PROJ_PATH2 {}'.format(os.environ['IND_PROJ_PATH2']))
# init IND_MODEL_PATH2
os.environ['IND_MODEL_PATH2'] = os.path.dirname(os.path.abspath(__file__))+'/model_dir'
print('Setting IND_MODEL_PATH2 {}'.format(os.environ['IND_MODEL_PATH2']))

import numpy as np
from mmcv import Config
from typing import Dict, List, Any, Optional

from indproj2.algos.utils.error_code import error_code_dict
from indproj2.algos.utils.pose_utils import get_group, get_pose
from indproj2.apis.inference import init_pipeline
from utils2.config_utils import get_default_config_path, fix_dict

from aoidata import DetPredictor, DetBox
# from indproj2.aoidata import DetPredictor, DetBox

# import logging
# FORMAT = '%(asctime)s %(levelname)s: %(message)s'
# logging.basicConfig(level=logging.INFO, filemode='w', format=FORMAT)

class AlgorithmInterfacePredictor(DetPredictor):

    def initialize(self, *, model_path: str, **kwargs: Dict[str, Any]) -> None:
        
        try:
            # pipeline
            self.config_path = get_default_config_path()
            self.config = Config.fromfile(self.config_path)
            self.config = fix_dict(self.config)
            self.pipeline = init_pipeline(self.config, gpu_id=0, verbose=True)

            # thresh
            self.THRESH_NAME = 'threshold.yaml'
            self.thresh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.THRESH_NAME)
            with open(self.thresh_path, 'r', encoding='utf8') as fp:
                raw_thr = yaml.load(fp, Loader=yaml.Loader)
            if isinstance(raw_thr, dict):
                self.thresh = raw_thr
            else:
                raise ValueError('Unknown format of threshold')
            
        except Exception:
            import traceback
            self.model = None
            self.thresh = None
            raise ValueError('failed to initialize detector', traceback.format_exc())

    @property
    def all_tags(self) -> List[str]:
        classes = []
        for module in self.config.pipeline.modules:
            if module.type in ['UltralyticsFasterDetector', 'MMFasterDetector']:
                classes = module['classes']
        return classes

    def inference(self, image: np.ndarray) -> List[DetBox]:
        
        fake_imageName = 'S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png'
        feed_dict = dict(images=[image], image_name=fake_imageName)
        pose = get_pose(feed_dict['image_name'], None)
        group = get_group(feed_dict['image_name'], None)
        feed_dict.update(dict(pose=pose, group=group))
        feed_dict['error_code'] = error_code_dict['success']
        feed_dict['error_reason'] = "success"
        results = self.pipeline(feed_dict=feed_dict)
        ret = []
        if 'preds' in results:
            preds = results['preds']
            for pred in preds:
                xyxy = pred['det_bbox']
                xywh = self.xyxy2xywh(xyxy)
                ret.append(DetBox(*tuple(float(v) for v in xywh),
                                score=float(pred['det_score']),
                                tag=pred['det_code'],
                                attributes=dict(is_ng=pred['det_score'] >= self.thresh[pred['det_code']])))
        return ret
    
    def inference_time_call(self, image: np.ndarray) -> List[DetBox]:

        fake_imageName = 'S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png'
        feed_dict = dict(images=[image], image_name=fake_imageName)
        pose = get_pose(feed_dict['image_name'], None)
        group = get_group(feed_dict['image_name'], None)
        feed_dict.update(dict(pose=pose, group=group))
        feed_dict['error_code'] = error_code_dict['success']
        feed_dict['error_reason'] = "success"
        results, time_dict = self.pipeline.time_call(feed_dict=feed_dict)
        ret = []
        if 'preds' in results:
            preds = results['preds']
            for pred in preds:
                xyxy = pred['det_bbox']
                xywh = self.xyxy2xywh(xyxy)
                ret.append(DetBox(*tuple(float(v) for v in xywh),
                                score=float(pred['det_score']),
                                tag=pred['det_code'],
                                attributes=dict(is_ng=pred['det_score'] >= self.thresh[pred['det_code']])))
        print(time_dict)
        return ret

    def xyxy2xywh(self, xyxy):
        bbox = np.array(xyxy)
        bbox[2:4] = bbox[2:4] - bbox[:2]
        return bbox
    
    def _set_device(self, device_type: str, device_id: int = 0) -> None:
        for key, _module in self.pipeline._modules.items():
            _module._set_device(device_type, device_id)

"""
python3 ./algorithm_interface_sdk.py /youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/val/NG/S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png
CUDA_VISIBLE_DEVICES=1,0 python3 ./algorithm_interface_sdk.py /youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/val/NG/S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png
"""


if __name__ == "__main__":
    import cv2
    interface = AlgorithmInterfacePredictor()
    interface.initialize(model_path=None)
    
    image_path = sys.argv[1]
    image_data = cv2.imread(image_path)

    print(interface.inference(image_data))
    interface.set_device('cuda', 0)
    print(interface.all_tags)

    # repeat_num = 500
    # start = time.time()
    # for i in range(repeat_num):
    #     interface.inference_time_call(image_data)
    # end = time.time()
    # print('average cost time is {}'.format((end-start)/repeat_num))
