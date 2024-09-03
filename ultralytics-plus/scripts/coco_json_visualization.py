import os
import json
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse

"""
COCO_JSON_PATH=/youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/annotations/train.json
PREFIX=/
SAVE_DIR=./vis/

python3 ./scripts/coco_json_visualization.py --coco_json_path $COCO_JSON_PATH --prefix $PREFIX --save_dir $SAVE_DIR
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the yolo')
    parser.add_argument('--coco_json_path', type=str, help='json_path')
    parser.add_argument('--prefix', type=str, help='coco json prefix')
    parser.add_argument('--save_dir', type=str, help='save dir', default='./vis/')
    args = parser.parse_args()
    
    coco_json_path = Path(args.coco_json_path)
    prefix = Path(args.prefix)
    save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    with coco_json_path.open('r') as f:
        json_data = json.load(f)
 
    images = json_data['images']
    annotations = json_data['annotations']

    imgid2annos = defaultdict(list)
    for anno_info in annotations:
        img_id = anno_info['image_id']
        imgid2annos[img_id].append(anno_info)

    for img_info in tqdm(images):
        img_id = img_info['id']
        img_path = prefix/img_info['file_name']
        annos = imgid2annos[img_id]
        img_data = cv2.imread(str(img_path))

        for anno_info in annos:
            x, y, w, h = anno_info['bbox']
            s, e = (x, y), (x+w, y+h)
            cv2.rectangle(img_data, s, e, (0, 0, 255), 1)
            cv2.putText(img_data, anno_info['category_name'], s, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        save_name = '|'.join(img_info['file_name'].split(os.path.sep))
        save_path = save_dir/save_name
        cv2.imwrite(str(save_path), img_data)
