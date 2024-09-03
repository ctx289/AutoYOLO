import cv2
import os
import json
import argparse
import shutil
from pathlib import Path
from utils import bbox_vis
from metric_validation import img_level_metrics


"""
COCO_JSON_PATH=/youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/annotations/val.json
PREDICTION_JSON_PATH=./runs/detect/val/prediction_val.json
PREFIX=/
SAVE_DIR=./runs/detect/val/img_level_validation

python3 ./scripts/img_level_validation.py --coco_json_path $COCO_JSON_PATH --prediction_json_path $PREDICTION_JSON_PATH\
      --prefix $PREFIX --save_dir $SAVE_DIR
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Test the model with vis')
    parser.add_argument('--coco_json_path', type=str, help='Ground truth annotations json path for validation.')
    parser.add_argument('--prediction_json_path', type=str, default='', help='The coco json used to predict the results, if not given will use val data in data of args.yaml')
    parser.add_argument('--prefix', type=str, help='Image prefix for gt coco json')
    parser.add_argument('--save_dir', type=str, help='The dir for validation results', default='./runs/detect/val/img_level_validation')
    parser.add_argument('--conf_json_path', type=str, help='Json file contains thresh of classes to evaluate', default=None)
    parser.add_argument('--omit_realrecall', action='store_true', help='Whether to omit saving real-recall imgs')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # custom
    coco_json_path = Path(args.coco_json_path)
    prediction_json_path = Path(args.prediction_json_path)
    prefix = Path(args.prefix)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  

    # conf
    if args.conf_json_path is not None:
        conf_json_path = Path(args.conf_json_path)
        with conf_json_path.open('r') as f:
            conf_dict = json.load(f)
    else:
        # Set the confidence threshold for all classes to 0.2 for default setting
        with open(args.coco_json_path, 'r') as f:
            content = json.load(f)
        conf_dict = dict()
        for item in content['categories']:
            conf_dict[item['name']] = 0.2

    # process
    lucky_recall_dir = save_dir/'lucky_recall'
    real_recall_dir = save_dir/'real_recall'
    miss_dir = save_dir/'miss'
    overkill_dir = save_dir/'overkill'

    if lucky_recall_dir.exists(): shutil.rmtree(lucky_recall_dir)
    if real_recall_dir.exists(): shutil.rmtree(real_recall_dir)
    if miss_dir.exists(): shutil.rmtree(miss_dir)
    if overkill_dir.exists(): shutil.rmtree(overkill_dir)

    lucky_recall_dir.mkdir(exist_ok=True)
    real_recall_dir.mkdir(exist_ok=True)
    miss_dir.mkdir(exist_ok=True)
    overkill_dir.mkdir(exist_ok=True)
    
    lucky_recall_infos, real_recall_infos, miss_img_infos, overkill_img_infos = img_level_metrics(coco_json_path, prediction_json_path, thresh_dict=conf_dict)

    # visualization
    for infos in lucky_recall_infos:
        img_info = infos['img_info']
        gt_bboxes = infos['gt_bboxes']
        pred_bboxes = infos['pred_bboxes']
        img_path = prefix/img_info['file_name']
        img_data = cv2.imread(str(img_path))
        for bbox_info in gt_bboxes:
            bbox = bbox_info[:4]
            cat_name = bbox_info[4]
            bbox_vis(img=img_data, bbox=bbox, color=(0, 255, 0), cat_name=cat_name)

        for bbox_info in pred_bboxes:
            bbox = list(map(int, bbox_info[:4]))
            cat_score = bbox_info[4]
            cat_name = bbox_info[5]
            bbox_vis(img=img_data, bbox=bbox, color=(0, 0, 255), cat_name=cat_name, cat_score=cat_score)

        save_path = lucky_recall_dir/('|'.join(img_info['file_name'].split(os.path.sep)))
        cv2.imwrite(str(save_path), img_data)

    # omit
    if not args.omit_realrecall:
        for infos in real_recall_infos:
            img_info = infos['img_info']
            gt_bboxes = infos['gt_bboxes']
            pred_bboxes = infos['pred_bboxes']
            img_path = prefix/img_info['file_name']
            img_data = cv2.imread(str(img_path))
            for bbox_info in gt_bboxes:
                bbox = bbox_info[:4]
                cat_name = bbox_info[4]
                bbox_vis(img=img_data, bbox=bbox, color=(0, 255, 0), cat_name=cat_name)

            for bbox_info in pred_bboxes:
                bbox = list(map(int, bbox_info[:4]))
                cat_score = bbox_info[4]
                cat_name = bbox_info[5]
                bbox_vis(img=img_data, bbox=bbox, color=(0, 0, 255), cat_name=cat_name, cat_score=cat_score)

            save_path = real_recall_dir/('|'.join(img_info['file_name'].split(os.path.sep)))
            cv2.imwrite(str(save_path), img_data)

    for infos in miss_img_infos:
        img_info = infos['img_info']
        gt_bboxes = infos['gt_bboxes']
        img_path = prefix/img_info['file_name']
        img_data = cv2.imread(str(img_path))
        for bbox_info in gt_bboxes:
            bbox = bbox_info[:4]
            cat_name = bbox_info[4]
            bbox_vis(img=img_data, bbox=bbox, color=(0, 255, 0), cat_name=cat_name)

        save_path = miss_dir/('|'.join(img_info['file_name'].split(os.path.sep)))
        cv2.imwrite(str(save_path), img_data)

    for infos in overkill_img_infos:
        img_info = infos['img_info']
        pred_bboxes = infos['pred_bboxes']
        img_path = prefix/img_info['file_name']
        img_data = cv2.imread(str(img_path))

        for bbox_info in pred_bboxes:
            bbox = list(map(int, bbox_info[:4]))
            cat_score = bbox_info[4]
            cat_name = bbox_info[5]
            bbox_vis(img=img_data, bbox=bbox, color=(0, 0, 255), cat_name=cat_name, cat_score=cat_score)

        save_path = overkill_dir/('|'.join(img_info['file_name'].split(os.path.sep)))
        cv2.imwrite(str(save_path), img_data)


if __name__ == "__main__":
    main()
