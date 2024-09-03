import argparse
import json
import re
import os
import imagesize
import copy
from pathlib import Path


labelme_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": 1024,
        "imageWidth": 1024
    }


def get_pose(image_path, product_name=""):
    """ get pose num for image path or image name

    Args:
        image_path (str): image_path

    Returns:
        int: pose
    """
    image_name = os.path.basename(image_path)
    pose = re.search(r"P(\d+)", image_name).group(1)
    return int(pose)


def generate_labelme_json_from_outers(outers, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key, value in outers.items():
        data = copy.deepcopy(labelme_data)
        absolute_jpg_path, outer = value
        suffix = Path(absolute_jpg_path).suffix
        data['imagePath'] = os.path.basename(absolute_jpg_path)
        image_width, image_height = imagesize.get(absolute_jpg_path)
        data['imageHeight'] = image_height
        data['imageWidth'] = image_width
        shape = {
            "label": "outer",
            "points": [],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
            }
        shape['points'] = [[outer[0], outer[1]], [outer[2], outer[3]]]
        data['shapes'].append(shape)
        with open(os.path.join(save_path, data['imagePath'].replace(suffix, '.json')), 'w') as f:
            json.dump(data, f, indent=4)
        print('======> save roi : ' + os.path.join(save_path, data['imagePath'].replace(suffix, '.json')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=str)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()

    train_jsons = args.train_json.split(',')
    outers = dict()
    for train_json in train_jsons:
        with open(train_json, 'r') as f:
            content = json.load(f)
        for image in content['images']:
            pose = get_pose(image['file_name'])
            width, height = image['width'], image['height']
            # check image 
            absolute_jpg_path = image['file_name']
            print(f'check {absolute_jpg_path}')
            if not os.path.exists(absolute_jpg_path):
                raise Exception(f"WARNING 'cannot load {absolute_jpg_path} for imagesize'")
            # check width and height
            image_width, image_height = imagesize.get(absolute_jpg_path)
            if width is not None and (width, height) != (image_width, image_height):
                raise Exception(f"WARNING 'The shape {(image_width, image_height)} of image does not match \
                    the height and weight {(width, height)} in labelme json'")
            # get outer
            if 'outer' in image:
                outer = image['outer']
            elif 'crop_info' in image:
                outer = image['crop_info']
            else:
                outer = [0, 0, width, height]
            # record outer
            if pose not in outers:
                outers[pose] = [absolute_jpg_path, outer]
    generate_labelme_json_from_outers(outers, args.save_path)
    

    
    