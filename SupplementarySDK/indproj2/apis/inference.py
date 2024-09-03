""" Inference
"""

import argparse
import json
import logging


from mmcv import Config

from ..algos.builder import build_pipeline
from ..algos.utils.pose_utils import get_pose, get_group
from ..algos.utils.error_code import error_code_dict
# from indproj2.algos.builder import build_pipeline
# from indproj2.algos.utils.pose_utils import get_pose, get_group
# from indproj2.algos.utils.error_code import error_code_dict


def init_pipeline(config, gpu_id, verbose=False):

    if isinstance(config, str):
        pipeline_cfg = Config.fromfile(config_path)
    else:
        pipeline_cfg = config

    if verbose:
        logging.info(json.dumps(dict(pipeline_cfg.pipeline), indent=4))

    pipeline = build_pipeline(pipeline_cfg.pipeline,
                              gpu_id=gpu_id,
                              verbose=verbose)
    return pipeline


# Deprecated
# data preparation is pre set in interface _prepare_feed_dict
def prepare_data(image_path):
    import cv2
    img = cv2.imread(image_path)
    
    # prepare feed_data
    feed_data = dict(images=[img], image_name=image_path)
    pose = get_pose(image_path, None)
    group = get_group(image_path, None)
    feed_data.update(dict(pose=pose, group=group))
    feed_data['error_code'] = error_code_dict['success']
    feed_data['error_reason'] = "success"

    return feed_data


if __name__ == '__main__':
    """
    python3 ./indproj2/apis/inference.py -c ./configs/liangang/base.py -i /youtu/xlab-team4/ryanwfu/24_second_sdk/625_dev/NG/S00040_C15_P15_L0_PI139_G1_M1_Y1_20230628200816.jpg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image-path',
                        type=str,
                        required=True,
                        help="image path to be inferenced")
    parser.add_argument('-c', '--config-path',
                        type=str,
                        required=True,
                        help="config path")
    parser.add_argument('--gpu-id', type=int, default=0, help="gpu id")
    args = parser.parse_args()
    config_path = args.config_path
    image_path = args.image_path
    gpu_id = args.gpu_id

    feed_dict = prepare_data(image_path)
    pipeline = init_pipeline(config_path, gpu_id, True)
    # 下一步结束,feed_dict 和 results中的内容一样
    result = pipeline(feed_dict)
    print(result)
