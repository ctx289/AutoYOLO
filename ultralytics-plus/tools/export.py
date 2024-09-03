import os
import time
import logging
import argparse
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralyticsp.custom import yaml_load, DictAction
from ultralyticsp.models import CustomYOLO, CustomRTDETR

def parse_args():
    parser = argparse.ArgumentParser(description='validate a trained detector')
    parser.add_argument('--config', help='train config file path, i.e. ./ultralyticsp/cfg/platform/yolov8n_default.yaml')
    parser.add_argument('--model', help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    parser.add_argument('--format', help='export format, i.e. onnx, tflite, torchscript', default='onnx')
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


def export(args, use_wrapper=True):
    """Runs YOLO model inference on input image(s)."""

    # cfg options
    if args.cfg_options is not None:
        cfg_options = args.cfg_options
    else:
        cfg_options = {}

    if use_wrapper:
         # Load a model
        cfg = get_cfg(args.config)
        if 'rtdetr' in os.path.basename(cfg.model):
            model = CustomRTDETR(args.model)
        elif 'yolo' in os.path.basename(cfg.model):
            model = CustomYOLO(args.model)
        else:
            raise NotImplementedError
        # Display model information (optional)
        model.info()
        # Export the model
        model.export(format=args.format, **cfg_options)


if __name__ == '__main__':
    args = parse_args()
    export(args)