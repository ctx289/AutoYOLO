""" Demo for inference data

Usage bash tools/inference iamge_path -m labelme (or will draw on image_path_draw.jpg)

The roi is tricky in this file
The old format of roi is
roi_dir
    - Gx
        - images
            - xxx.jpg
            - xxx.json

The new format is
roi_dir
    - xxx.jpg
    - xxx.json

So we need to check the format of roi in get_roi function

"""

import argparse
import copy
import io
import json
import logging
import os
import re
import shutil
from code import interact
from collections import defaultdict
from distutils.log import INFO
from pathlib import Path

import cv2
import numpy as np
from indproj2.algos.utils.pose_utils import (get_group, get_pose,
                                            get_product_pose)
from interface2.algorithm_interface_standard import AlgorithmInterface
from tqdm import tqdm

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, filemode='w', format=FORMAT)

global interface
interface = None

# information
INFO_COLLECTER = defaultdict(lambda: defaultdict(int))


def demo_image(image_path,
               mode='labelme',
               output_dir=None,
               verbose=True,
               time_check=False,
               draw_advanced_mode=False):
    global interface
    # preparea input
    image_data = cv2.imread(image_path)
    if image_data is None:
        raise ValueError("Loading image {} failed".format(image_path))

    request = {"images": [{"image_name": image_path, "image_data": image_data}]}
    response = interface.SinglePosePredict(request, time_statistic=time_check)
    if verbose:
        print("========== printing results =================")
        print(json.dumps(response))

    if time_check and verbose:
        print("Time of each stage ==========================")
        for key in response['time_stage']:
            print("  {}: {}".format(key, response['time_stage'][key]))

    labelme_map = {}
    labelme_map["flags"] = {}
    labelme_map["shapes"] = []
    labelme_map["version"] = "4.5.6"
    labelme_map["imageData"] = None
    labelme_map["imageHeight"] = response["image_height"]
    labelme_map["imageWidth"] = response["image_width"]
    labelme_map["imagePath"] = response["imageName"].split("/")[-1]

    for result in response["result"]:
        shape = {}
        shape["flags"] = {}
        shape["group_id"] = None
        shape["shape_type"] = "rectangle"
        shape["label"] = result['code'] + "-" + str(format(result['score'], '.4f'))
        shape["points"] = [[result["bbox"][0], result["bbox"][1]],
                           [result["bbox"][2], result["bbox"][3]]]
        labelme_map["shapes"].append(shape)

    # draw result
    state = 0
    for res in response["origin_info"].get('preds', []):
        if 'filter_by' not in res:
            res['filter_by'] = 'high_ng'
        shape = {}
        shape["flags"] = {}
        shape["group_id"] = None
        shape["shape_type"] = "rectangle"
        shape["label"] = "{}-{}-{}-{}-{}".format(
            res['det_code'], round(res['det_score'], 4), res['area'],
            round(res.get('real_area', 0.0), 4), res['filter_by'])

        shape["points"] = [[res["det_bbox"][0], res["det_bbox"][1]],
                           [res["det_bbox"][2], res["det_bbox"][3]]]
        labelme_map["shapes"].append(shape)

        # draw segmentation if has
        if 'polygon' in res.keys() and res['polygon'] is not None and len(
                res['polygon']):
            shape = {}
            shape["flags"] = {}
            shape["group_id"] = None
            shape["shape_type"] = "polygon"
            shape["label"] = "polygon-" + res['det_code'] + "-" + str(
                res['det_score']) + "-" + str(res['area']) + "-" + str(
                    res['filter_by'])
            polygon = res['polygon']
            polygon = [item for pol in polygon for item in pol]
            polygon = np.array(polygon).reshape((-1, 2)).tolist()
            shape["points"] = polygon
            labelme_map["shapes"].append(shape)

        # draw ensemble if has
        if 'second_codes' in res.keys():
            for idx in range(len(res['second_codes'])):
                shape = {}
                shape["flags"] = {}
                shape["group_id"] = None
                shape["shape_type"] = "rectangle"
                shape["label"] = "second-" + res["second_codes"][
                    idx] + "-" + str(format(res['second_scores'][idx], '.4f'))
                box = res['second_bboxes'][idx]
                shape["points"] = [[box[0], box[1]], [box[2], box[3]]]
                labelme_map["shapes"].append(shape)
    
    if mode == "draw":
        if output_dir is None:
            draw_path = image_path.replace('.jpg', '-draw.jpg')
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_name = os.path.basename(image_path).replace(
                '.jpg', '-draw.jpg')
            draw_path = os.path.join(output_dir, image_name)
        
        if draw_advanced_mode and output_dir is not None:
            subfolders = ['NG', 'OK', 'FAIL', 'BLOCK_POSE']
            for subfolder in subfolders:
                if not os.path.exists(os.path.join(output_dir, subfolder)):
                    os.makedirs(os.path.join(output_dir, subfolder))
            image_name = os.path.basename(image_path).replace('.jpg', '-draw.jpg')
            if len(response['result']) != 0:
                draw_path = os.path.join(output_dir, 'NG', image_name)
                state = 'NG'
            elif len(response['result']) == 0 and 'preds' in response["origin_info"].keys():
                if len(response["origin_info"]['preds']) != 0:
                    draw_path = os.path.join(output_dir, 'OK', image_name)
                    state = 'OK'
                    return response['time_stage'], state
                else:
                    draw_path = os.path.join(output_dir, 'FAIL', image_name)
                    state = 'FAIL'
            else:
                draw_path = os.path.join(output_dir, 'BLOCK_POSE', image_name)
                state = 'BLOCK_POSE'
                return response['time_stage'], state

        image = cv2.imread(image_path)

        for res in response['origin_info']['preds']:
            x1 = res["det_bbox"][0]
            y1 = res["det_bbox"][1]
            x2 = res["det_bbox"][2]
            y2 = res["det_bbox"][3]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # label = res['det_code'] + "_" + str(
            #     format(res['det_score'], '.4f')) + "_" + res.get("filter_by", "")
            label = res['det_code'] + "_" + str(
                format(res['det_score'], '.2f'))
            cv2.putText(image, label, (x1, y1 + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)

        # for res in response['result']:
        #     x1 = res["bbox"][0]
        #     y1 = res["bbox"][1]
        #     x2 = res["bbox"][2]
        #     y2 = res["bbox"][3]
        #     image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #     label = res['code'] + "_" + str(format(res['length'], '.1f'))
        #     cv2.putText(image, label, (x1, y1 + 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
            
        cv2.imwrite(draw_path, image)
        if verbose:
            print("Image is saved to {}".format(draw_path))
    elif mode == "labelme":
        output_path = ""
        if output_dir is None:
            output_path = image_path.replace('.jpg', '.json')
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, image_name.replace('.jpg', '.json'))
            shutil.copy(
                image_path, os.path.join(output_dir, image_name)
            )
        with open(output_path, "w") as f:
            f.write(json.dumps(labelme_map, indent=4))
        if verbose:
            print("Image is saved to {}".format(output_path))
    else:
        pass

    return response['time_stage'], state


def demo_folder(input_dir,
                mode='labelme',
                output_dir=None,
                time_check=False,
                verbose=True,
                draw_advanced_mode=False):

    time_collect = []
    analysis = {'NG':0, 'OK':0, 'FAIL':0, 'BLOCK_POSE':0}
    for root, dirs, files in tqdm(os.walk(input_dir)):
        for img_name in files:
            if img_name[-3:] != "jpg" or len(img_name.split("_")) < 3:
                continue

            image_path = os.path.join(input_dir, root, img_name)
            if not os.path.exists(image_path):
                raise ValueError(
                    "{} does not exists, note: input dir must be full path not relative path".format(
                        image_path))
            
            time_stage, state = demo_image(image_path,
                                            mode,
                                            output_dir,
                                            verbose=verbose,
                                            time_check=time_check,
                                            draw_advanced_mode=draw_advanced_mode
                                            )
            time_collect.append(time_stage)
            if state!=0:
                analysis[state] += 1
                print(analysis)
    if time_check and len(time_collect):
        for key in time_collect[0].keys():
            print("{}: {}".format(key, np.average([x[key] for x in time_collect])))


def demo_coco_json(input_coco_json,
                mode='labelme',
                output_dir=None,
                time_check=False,
                verbose=True,
                draw_advanced_mode=False):

    with open(input_coco_json, "r") as f:
        json_data = json.load(f)
    if 'images' in json_data.keys():
        time_collect = []
        analysis = {'NG':0, 'OK':0, 'FAIL':0, 'BLOCK_POSE':0}
        for image_item in json_data['images']:
            image_path = image_item['file_name']
            if not os.path.exists(image_path):
                raise ValueError(
                    "{} does not exists, note: input dir must be full path not relative path".format(
                        image_path))
            time_stage, state = demo_image(image_path,
                                            mode,
                                            output_dir,
                                            verbose=verbose,
                                            time_check=time_check,
                                            draw_advanced_mode=draw_advanced_mode
                                            )
            time_collect.append(time_stage)
            if state!=0:
                analysis[state] += 1
                print(analysis)
        if time_check and len(time_collect):
            for key in time_collect[0].keys():
                print("{}: {}".format(key, np.average([x[key] for x in time_collect])))

"""
# liangang
python3 ./tools/demo/demo.py ../625_dev/NG/S00013_C15_P15_L0_PI139_G1_M1_Y1_20230628200302.jpg -c ./configs/liangang/base.py -o ../625_dev/NG_Output/ -m draw
python3 ./tools/demo/demo.py ../625_dev/NG/ -c ./configs/liangang/base.py -o ../625_dev/NG_Output/ -m draw --draw-advanced-mode
python3 ./tools/demo/demo.py ../625_dev/OK/ -c ./configs/liangang/base.py -o ../625_dev/OK_Output/ -m draw --draw-advanced-mode
python3 ./tools/demo/demo.py ../625_dev/other_pose/ -c ./configs/liangang/base.py -o ../625_dev/other_pose_Output/ -m draw --draw-advanced-mode
python3 ./tools/demo/demo.py /youtu/xlab-team2/tivp_v141_new/tmp/7404/inputs/all_test.json -c ./configs/liangang/base.py -o ../625_dev/all_test_output/ -m draw --draw-advanced-mode
# YZ_BOTTLECAP1
python3 ./tools/demo/demo.py /youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/val/NG/S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png\
      -c ./configs/YZ_BOTTLECAP1/bisenetv1_rtdetr.py -o ./ -m draw
python3 ./tools/demo/demo.py /youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/val/NG/S00281_C02_P003_L0_PI84_G1_M1_20230711032203.png\
      -c ./configs/YZ_BOTTLECAP1/bisenetv1_yolov8n.py -o ./ -m draw
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        type=str,
                        help="Input can be a image or a folder")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help=
        "Output dir if not specified, will draw under the same folder of input"
    )
    parser.add_argument('-c',
                        '--config-path',
                        type=str,
                        default=None,
                        required=False,
                        help="config path if not specified will use default")

    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        choices=["none", 'labelme', 'draw'],
                        default="none",
                        help="Mode must in labelme or draw")
    
    parser.add_argument("--draw-advanced-mode",
                        action='store_true',
                        help="Whether check time of each stage")

    parser.add_argument("--time-check",
                        action='store_true',
                        help="Whether check time of each stage")
    args = parser.parse_args()

    interface = AlgorithmInterface(config_path=args.config_path)

    input_path = os.path.realpath(args.input)
    mode = args.mode
    time_check = args.time_check
    draw_advanced_mode = args.draw_advanced_mode

    output_dir = args.out
    if os.path.isdir(input_path):
        demo_folder(
            input_path, mode, output_dir,
            time_check=time_check, draw_advanced_mode=draw_advanced_mode)
    elif input_path.endswith('.json'):
        demo_coco_json(
            input_path, mode, output_dir,
            time_check=time_check, draw_advanced_mode=draw_advanced_mode)
    else:
        demo_image(input_path, mode, output_dir,
                   time_check=time_check, draw_advanced_mode=draw_advanced_mode)

    print("Finished.")
    if INFO_COLLECTER:
        print("INFOS: ", INFO_COLLECTER)
