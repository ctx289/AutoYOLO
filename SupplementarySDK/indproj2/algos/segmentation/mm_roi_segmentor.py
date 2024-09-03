""" Define the jb roi segmentor """
from typing_extensions import runtime
from ..builder import MODULES
from .base_segmentor import BaseSegmentor
import cv2
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import logging
from ..utils.pose_utils import get_pose, get_product_pose

@MODULES.register_module()
class MMRoiSegmentor(BaseSegmentor):
    
    def __init__(self,
                 config,
                 ckpt,
                 classes={0:'bg', 1:'outer'},
                 get_polygon=True,
                 input_name='images',
                 output_name='preds',
                 block_poses=[],
                 use_product_pose=False,
                 no_preds_skip_roi_seg=False,
                 gpu_id=0,
                 verbose=False,
                 **kwargs):
        super(MMRoiSegmentor, self).__init__(config, 
                                            ckpt, 
                                            input_name=input_name, 
                                            output_name=output_name, 
                                            gpu_id=gpu_id,
                                            verbose=verbose,
                                            **kwargs)    
        self.get_polygon = get_polygon
        self.use_product_pose = use_product_pose
        self.block_poses = block_poses
        self.no_preds_skip_roi_seg = no_preds_skip_roi_seg
        self.classes = OrderedDict(classes)
        self.classes_rev = {v: k for k, v in classes.items()}
    
    def init(self, config, ckpt, gpu_id):
        from mmseg.apis import inference_segmentor, init_segmentor
        self.inference_segmentor = inference_segmentor
        if self.verbose:
            logging.info("Initializing ROISegmentor on cuda:{}".format(gpu_id))
        return init_segmentor(config, ckpt, device=f"cuda:{gpu_id}")
    
    def _set_device(self, device_type: str, device_id: int = 0) -> None:
        print(f'set MMRoiSegmentor device to {device_type}:{device_id}')
        self.model.to(f'{device_type}:{device_id}')
    
    @staticmethod
    def _get_pose(image_name=None, use_product_pose=False, feed_dict=None):
        if feed_dict is None and image_name is None:
            raise ValueError(
                "Image name and feed_dict cannot both be None for getting pose."
            )
        if use_product_pose:
            if feed_dict is not None:
                return feed_dict.get('product_pose', get_product_pose(feed_dict['image_name']))
            else:
                return get_product_pose(image_name)
        else:
            if feed_dict is not None:
                return feed_dict.get('pose', get_pose(feed_dict['image_name']))
            else:
                return get_pose(image_name)

    def predict(self, feed_dict):
        # process
        try:    
            imgs = feed_dict[self.input_name]
            outputs = self.inference_segmentor(self.model, imgs[0])
            mask = outputs[0]    
            roi_mask = mask.astype(np.uint8)    # value 0 / 1 
            if self.get_polygon:
                roi_mask_for_polygon = deepcopy(roi_mask)
                contours, _ = cv2.findContours(roi_mask_for_polygon, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
                roi_polygon = [i.flatten().tolist() for i in contours]
            else:
                roi_polygon = None
        except:
            return False, None, None
        
        return True, roi_mask, roi_polygon

    def __call__(self, feed_dict, **kwargs):
        """ inference with feed data

        Args:
            feed_dict (dict)
        """

        pose = self._get_pose(feed_dict=feed_dict,
                              use_product_pose=self.use_product_pose)

        # 1. Check block pose
        if pose in self.block_poses:
            if "roi_seg" in feed_dict.keys():
                return feed_dict
            feed_dict['roi_seg'] = {
                'roi_seg_ok': False,
                'roi_mask': None,
                'roi_polygon': None,
                'roi_seg_msg': "ROI-Seg-Skip-BlockPose."
            }
            return feed_dict
        
        # 2. set default roi_seg
        feed_dict['roi_seg'] = {
            'roi_seg_ok': False,
            'roi_mask': None,
            'roi_polygon': None,
            'roi_seg_msg': 'default_msg'
        }
        
        # 3. No preds skip
        if self.no_preds_skip_roi_seg:
            if self.output_name in feed_dict and len(feed_dict[self.output_name]) == 0:
                feed_dict['roi_seg']['roi_seg_msg'] = 'No-Preds-Skip-Roi-Seg'
                return feed_dict

        # 4. ROI seg process
        roi_seg_ok, roi_mask, roi_polygon = self.predict(feed_dict)
        if roi_seg_ok:
            feed_dict['roi_seg']['roi_seg_ok'] = True
            # feed_dict['roi_seg']['roi_mask'] = roi_mask
            feed_dict['roi_seg']['roi_mask'] = None
            feed_dict['roi_seg']['roi_polygon'] = roi_polygon
            feed_dict['roi_seg']['roi_seg_msg'] = 'ROI-Seg-Successed'
        else:
            feed_dict['roi_seg']['roi_seg_msg'] = 'ROI-Seg-Failed'

        return feed_dict
