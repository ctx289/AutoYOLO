import json
import cv2
import os
import copy
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from utils import preds_vis, gts_vis

from ultralytics.cfg import get_cfg

from ultralyticsp.custom import yaml_load
from ultralyticsp.models import CustomYOLO, CustomRTDETR

"""
MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default/train
SAVE_DIR=./runs/detect/val

python3 ./scripts/get_val_visualization.py --config $MODELCONFIG --model $WORKDIR/weights/best.pt --data $DATACONFIG --save_dir $SAVE_DIR
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Test the model with vis')
    parser.add_argument('--config', help='train config file path, i.e. ./ultralyticsp/cfg/platform/yolov8n_default.yaml')
    parser.add_argument('--model', type=str, help='ckpt_path')
    parser.add_argument('--data', help='path to data file, i.e. ./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml')
    parser.add_argument('--save_dir', type=str, help='save dir', default='./runs/detect/val')
    parser.add_argument('--conf', type=float, help='conf for vis', default=0.1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # custom data cfg
    data_cfg = yaml_load(args.data)
    coco_json_path = Path(os.path.join(data_cfg['path'], data_cfg['val']))
    prefix = Path(data_cfg['val_prefix'])
    assert data_cfg['coco_format_json']
    roi_crop = data_cfg['roi_crop']
    roi_key = data_cfg['roi_key']
    classes = list(data_cfg['names'].values())

    # init model
    cfg = get_cfg(args.config)
    if 'rtdetr' in os.path.basename(cfg.model):
        model = CustomRTDETR(args.model)
    elif 'yolo' in os.path.basename(cfg.model):
        model = CustomYOLO(args.model)
    else:
        raise NotImplementedError

    # prepare data
    with coco_json_path.open('r') as f:
        coco_data = json.load(f)
    images = coco_data['images']
    annotations = coco_data['annotations']
    imgid2annos = defaultdict(list)

    for anno_info in annotations:
        img_id = anno_info['image_id']
        imgid2annos[img_id].append(anno_info)

    # process
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for img_info in tqdm(images):
        img_id = img_info['id']
        annos = copy.deepcopy(imgid2annos[img_id])
        img_path = prefix/img_info['file_name']
        img_data = cv2.imread(str(img_path))
        if roi_crop:
            roi_info = img_info[roi_key]
            rx1, ry1, rx2, ry2 = roi_info
            img_data = img_data[ry1:ry2, rx1:rx2]
            for anno_info in annos:
                anno_info['bbox'] = (np.array(anno_info['bbox']) - np.array([rx1, ry1, 0, 0])).tolist()
                if 'bbox_xyxy' in anno_info:
                    anno_info['bbox_xyxy'] = (np.array(anno_info['bbox_xyxy']) - np.array([rx1, ry1, rx1, rx2])).tolist()
                if 'segmentation' in anno_info:
                    anno_info['segmentation'][0][::2] = (np.array(anno_info['segmentation'][0][::2]) - rx1).tolist()
                    anno_info['segmentation'][0][1::2] = (np.array(anno_info['segmentation'][0][1::2]) - ry1).tolist()

        preds = model.predict(img_data, conf=args.conf, verbose=False)[0]

        img_data = gts_vis(img=img_data, annos=annos, only_keep_cat=True, keep_cat=classes)
        img_data = preds_vis(img=img_data, preds=preds, only_keep_cat=True, keep_cat=classes)

        save_name = '|'.join(img_info['file_name'].split(os.path.sep))
        save_path = save_dir/save_name
        cv2.imwrite(str(save_path), img_data)


if __name__ == "__main__":
    main()