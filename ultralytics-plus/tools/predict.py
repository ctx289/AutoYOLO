# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import argparse
from ultralytics.cfg import get_cfg
from ultralytics import settings
from tools.dictaction import DictAction

from ultralyticsp.custom import yaml_load
from ultralyticsp.models import CustomYOLO, CustomRTDETR
from ultralyticsp.models.yolo.detect import COCOYOLODetectionPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='validate a trained detector')
    parser.add_argument('--config', help='train config file path, i.e. ./ultralyticsp/cfg/platform/yolov8n_default.yaml')
    parser.add_argument('--model', help='path to model file, i.e. best.pt, represents the trained ckpt')
    parser.add_argument('--source', help='source image to predict, i.e. ./assets/S00265_C02_P003_L0_PI84_G1_M1_20230711032203.png')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def predict(args, use_wrapper=True):
    """Runs YOLO model inference on input image(s)."""

    # cfg options
    if args.cfg_options is not None:
        cfg_options = args.cfg_options
    else:
        cfg_options = {}

    if use_wrapper:
        """
        # save_dir: None / no save
        """
        settings.update({'runs_dir': './runs'})
        cfg_options = {**cfg_options, **{'exist_ok':True}}
        cfg = yaml_load(args.config)
        if 'rtdetr' in os.path.basename(cfg['model']):
            # When predicting, there is no difference between RTDETR and customRTDETR
            model = CustomRTDETR(args.model)
        elif 'yolo' in os.path.basename(cfg['model']):
            # When predicting, there is no difference between YOLO and customYOLO
            model = CustomYOLO(args.model)
        else:
            raise NotImplementedError
        # Display model information (optional)
        model.info()
        # Default conf is 0.25 for predict if not specified
        results = model.predict(source=args.source, save=True, **cfg_options)
    else:
        """
        # === deprecated ===
        # 不推荐使用, 需额外输入args.config进行初始化, 文件保存路径通过输入的args.config中的project指定
        # 当 args.project 不存在时, 会通过Ultralytics Settings指定路径(第二选择), 参考ultralytics/docs/quickstart.md, 但支持Modifying Settings
        # Ultralytics Settings 路径保存目前在/root/.config/Ultralytics/settings.yaml
        # example: args.project is None: Results saved to /youtu/xlab-team4/ryanwfu/26_AutoModels/AutoProject/runs/detect/predict7
        # Attention - onnx 格式的模型可以在下面的api中使用
        """
        simple_cfg = {'mode':'predict', 'exist_ok':True}
        simple_cfg = {**simple_cfg, **cfg_options}
        predictor = COCOYOLODetectionPredictor(cfg=args.config, overrides=simple_cfg)
        results = predictor(source=args.source, model=args.model)
    
    print(results)


if __name__ == '__main__':
    args = parse_args()
    predict(args)
