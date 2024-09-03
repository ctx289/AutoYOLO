import json
import os
import sys
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class AutoResolution():

    def __init__(self, coco_json_path:[str or list(str)], scaled_bbox_area_factor=42, enable_pose=True):
        """
        input
        @coco_json_path: the path to coco json, str or list[coco_json1, coco_json2...].
        @scaled_bbox_area_factor: the scale factor of optimal scaled annotation bbox, default is set to 42.
        """
        self.coco_path = coco_json_path
        self.default_res = 800
        self.enable_pose = enable_pose
        self.scaled_bbox_area_factor = scaled_bbox_area_factor
        self.scaled_bbox_area = scaled_bbox_area_factor * scaled_bbox_area_factor
        self.recommend_resolution = (self.default_res, self.default_res)
        self.pose_recommend_resolution = {}
    
    def get_pose(self, image_path, product_name=""):
        """ get pose num for image path or image name
        Args:
            image_path (str): image_path
        Returns:
            int: pose
        """
        image_name = os.path.basename(image_path)
        pose = re.search(r"P(\d+)", image_name).group(1)
        return int(pose)

    def load_coco_jsons(self, coco_json_paths):
        if isinstance(coco_json_paths, str):
            assert len(coco_json_paths) > 0 and coco_json_paths.endswith('json')
            try:
                with open(coco_json_paths, 'r') as f:
                    coco_content = json.load(f)
                logger.info(f"[Auto Resolution] load {coco_json_paths}")
            except:
                raise ValueError(f"[Auto Resolution] load {coco_json_paths} failed")

        elif isinstance(coco_json_paths, list):
            coco_datas = []
            for coco_json_path in coco_json_paths:
                try:
                    with open(coco_json_path, 'r') as f:
                        coco_data_ = json.load(f)
                    coco_datas.append(coco_data_)
                    logger.info(f"[Auto Resolution] load {coco_json_path}")
                except:
                    raise ValueError("[Auto Resolution] load {coco_json_path} failed")
            coco_content = self.combine_alist_coco_json(coco_datas)
        else:
            raise ValueError("[Auto Resolution] only str and list are supported in coco json.")
        return coco_content
    
    def combine_alist_coco_json(self, coco_datas):
        """Combine a list of coco json into one
        """
        img_counter = 0
        anno_counter = 0
        images_all = []
        annotations_all = []
        categories = []
        
        for coco_data in coco_datas:
            img_id_anno = defaultdict(list)
            for anno in coco_data['annotations']:
                img_id_anno[anno['image_id']].append(anno)

            img_id_info = dict()
            for img_info in coco_data['images']:
                img_id_info[img_info['id']] = img_info
                
            for img_id, img_info in img_id_info.items():
                annos = img_id_anno[img_id]
                img_info['id'] = img_counter
                for anno in annos:
                    anno['image_id'] = img_counter
                    anno['id'] = anno_counter
                    anno_counter += 1
                img_counter += 1
                images_all.append(img_info)
                annotations_all.extend(annos)

            # get categories
            if len(coco_data['categories']) > len(categories):
                categories = coco_data['categories']

        content = dict()
        content['images'] = images_all
        content['annotations'] = annotations_all
        content['categories'] = categories

        return content
    
    def trim_mean(self, lst, percent=0.3):
        n = int(len(lst) * percent / 2)
        lst = sorted(lst)
        lst = lst[n:len(lst)-n]
        return sum(lst) / len(lst) 
        
    def process(self, coco_content):
        img_infos = coco_content['images']
        ann_infos = coco_content['annotations']
        
        img_hs = []
        img_ws = []
        bbox_areas = []
        
        img_hs_pose = defaultdict(list)
        img_ws_pose = defaultdict(list)
        bbox_areas_pose = defaultdict(list)
        img_id_pose_dict = dict()

        # Statistical image width and height
        for img_info in img_infos:
            if self.enable_pose:
                pose_id = self.get_pose(img_info['file_name'])
                img_id_pose_dict[img_info['id']] = pose_id
            
            if 'crop_info' in img_infos[0].keys():
                _x1, _y1, _x2, _y2 = img_info['crop_info']
                x1, x2 = min(_x1, _x2), max(_x1, _x2)
                y1, y2 = min(_y1, _y2), max(_y1, _y2)
                img_hs.append(y2 - y1)
                img_ws.append(x2 - x1)
                if self.enable_pose:
                    img_hs_pose[pose_id].append(y2 - y1)
                    img_ws_pose[pose_id].append(x2 - x1)
            else:
                img_hs.append(img_info['height'])
                img_ws.append(img_info['width'])
                if self.enable_pose:
                    img_hs_pose[pose_id].append(img_info['height'])
                    img_ws_pose[pose_id].append(img_info['width'])

        total_ave_img_h = np.mean(np.array(img_hs))
        total_ave_img_w = np.mean(np.array(img_ws))
        if self.enable_pose:
            pose_ave_img_h = {k: np.mean(np.array(v)) for k, v in img_hs_pose.items()}
            pose_ave_img_w = {k: np.mean(np.array(v)) for k, v in img_ws_pose.items()}
        
        # Count the average area of box
        for ann in ann_infos:
            w, h = ann['bbox'][2:]
            area_ = w * h
            bbox_areas.append(area_)
            if self.enable_pose:
                bbox_areas_pose[img_id_pose_dict[ann['image_id']]].append(area_)

        total_ave_ann_area = self.trim_mean(bbox_areas)
        if self.enable_pose:
            pose_ave_ann_area = {k: self.trim_mean(v) for k, v in bbox_areas_pose.items()}
        
        # Calculate recommended resolution
        tmp_resolution = self.calculate_recommended_resolution(total_ave_img_h, total_ave_img_w, total_ave_ann_area)
        self.recommend_resolution = self.get_32_resolution_clip(tmp_resolution)
        if self.enable_pose:
            for pose_id, pose_ann_area in pose_ave_ann_area.items():
                tmp_resolution = self.calculate_recommended_resolution(pose_ave_img_h[pose_id], pose_ave_img_w[pose_id], pose_ann_area)
                self.pose_recommend_resolution[pose_id] = self.get_32_resolution_clip(tmp_resolution)
    
    def calculate_recommended_resolution(self, h=0, w=0, ann_ave_area=0):
        """
        Formular: 
            s.t: ScaleImgSize * BBoxsize / ImgSize > scaled_bbox_area_factor*scaled_bbox_area_factor
        Input:
        @h: the input hight of ScaleImgSize
        @w: the input width of ScaleImgSize
        @ann_ave_area: the ave area of annotated bbox in given coco json.
        return:
        optimal img shape(h,w)
        """
        # if ann_ave_area > self.scaled_bbox_area:
        ann_ave_area_factor = np.sqrt(ann_ave_area)
        scaled_bbox_area_factor = self.get_scale_bbox_factor(h, w, ann_ave_area)
        shape_s_h = int(scaled_bbox_area_factor / ann_ave_area_factor * h)
        shape_s_w = int(scaled_bbox_area_factor / ann_ave_area_factor * w)
        return (shape_s_h, shape_s_w) 
    
    def get_scale_bbox_factor(self, h, w, ann_ave_area):
        if ann_ave_area < 300 and min(h, w) > 4800:
            # the ann is small and the image is large
            return 5
        elif ann_ave_area < 1000 and ann_ave_area > 500 and min(h, w) > 3600:
            # the ann is normal and image is big
            return 17
        elif ann_ave_area < 100 and ann_ave_area > 50 and min(h, w) < 1200:
            # the image is small and the ann is tiny 
            return 7 
        else:
            return self.scaled_bbox_area_factor
    
    def get_32_resolution_clip(self, resolution):
        res = max(resolution)
        if res % 32 != 0:
            res = (res // 32 + 1) * 32
        return min(max(res, 800), 1120)
    
    def get_recommended_resolution(self) -> (int, dict):
        try:
            coco_content = self.load_coco_jsons(self.coco_path)
            self.process(coco_content)
        except Exception as e:
            logger.warning(f"[Auto Resolution] error: {e}!")
            return self.default_res, {}
        return self.recommend_resolution, self.pose_recommend_resolution

if __name__ == '__main__':
    coco_json_path = sys.argv[1]
    if os.path.isdir(coco_json_path):
        coco_json_name = os.listdir(coco_json_path)
        coco_json_ = [os.path.join(coco_json_path, c) for c in coco_json_name if c.endswith(".json")]
        coco_json_path = coco_json_
    
    AutoRES = AutoResolution(coco_json_path, enable_pose=False)
    recommend_resolution, pose_recommend_resolution = AutoRES.get_recommended_resolution()
    print(recommend_resolution)
    print(pose_recommend_resolution)
