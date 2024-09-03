import argparse
import json
import re
import os
import yaml
from mmcv import Config


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pipeline_config_path", type=str)
    parser.add_argument("--output_pipeline_config_path", type=str)
    parser.add_argument("--det_arch", type=str, default='UltralyticsFasterDetector', choices=['PPFasterDetector', 'UltralyticsFasterDetector'])
    parser.add_argument("--det_work_dir", type=str)
    parser.add_argument("--roi_dir", type=str, default=None)
    args = parser.parse_args()

    # 1. scan work dir    
    # print(f'======> scan {args.det_work_dir}')
    if args.det_arch == 'PPFasterDetector':
        assert os.path.exists(args.det_work_dir)
        work_name = os.path.basename(args.det_work_dir).split('.yml')[0]
        det_config = os.path.join(args.det_work_dir, 'output_inference', work_name, 'infer_cfg.yml')
        det_ckpt = os.path.join(args.det_work_dir, 'output_inference', work_name+'/')
        with open(det_config, "r") as file:
            data = yaml.safe_load(file)
        det_classes = data['label_list']
    elif args.det_arch == 'UltralyticsFasterDetector':
        assert os.path.exists(args.det_work_dir)
        det_config = os.path.abspath(os.path.join(args.det_work_dir, 'args.yaml'))
        det_ckpt = os.path.abspath(os.path.join(args.det_work_dir, 'weights/best.pt'))
        det_classes = None
    else:
        raise NotImplementedError

    # 2. parse pipeline config
    # print(f'======> parse {args.input_pipeline_config_path}')
    cfg = Config.fromfile(args.input_pipeline_config_path)
    for i, module in enumerate(cfg['pipeline']['modules']):
        if i == 0:
            # det
            module['type'] = args.det_arch
            module['config'] = det_config
            module['ckpt'] = det_ckpt
            module['classes'] = det_classes
            module['crop_by_outer'] = True if args.roi_dir else False
            module['roi_dir'] = args.roi_dir
    
    # 3. save
    if not os.path.exists(os.path.dirname(args.output_pipeline_config_path)):
        os.makedirs(os.path.dirname(args.output_pipeline_config_path))
    cfg.dump(args.output_pipeline_config_path)
    # print(f'======> save to {args.output_pipeline_config_path}')
