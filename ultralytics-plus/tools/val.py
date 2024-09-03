import os
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from tools.dictaction import DictAction
from tools.dictaction import ModelLoader, FileModelLoader


def parse_args():
    parser = argparse.ArgumentParser(description='validate a trained detector')
    parser.add_argument('--config', help='train config file path, i.e. ./ultralyticsp/cfg/platform/yolov8n_default.yaml')
    parser.add_argument('--model', help='path to pretrained model, i.e. best.pt, represents the trained ckpt')
    parser.add_argument('--data', help='path to data file, i.e. ./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
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
    parser.add_argument('--format-only', action='store_true', help='Whether to inference format-only')
    parser.add_argument('--jsonfile-prefix', type=str, help='jsonfile prefix', default='./runs/detect/val/test')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


def val(args, use_wrapper=True, model_loader: ModelLoader = None):
    """
    一般不指定model_loader
    当未指定model_loader, args.config 中 model 字段表示模型方法, args.model 表示训练好的模型;
    """

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    # Normal Import
    from ultralyticsp.models.yolo.detect import COCOYOLODetectionValidator
    from ultralyticsp.models import CustomYOLO, CustomRTDETR
    from ultralyticsp.custom import yaml_load
    from ultralytics.cfg import get_cfg
    from ultralytics import settings

    # cfg options
    if args.cfg_options is not None:
        cfg_options = args.cfg_options
    else:
        cfg_options = {}
    
    if args.work_dir is not None:
        cfg_options['project'] = os.path.dirname(args.work_dir)
        cfg_options['name'] = os.path.basename(args.work_dir)

    # load binary stream
    if model_loader is None:
        model_loader = FileModelLoader(args.model)
    model_Bytes = model_loader.load()

    if use_wrapper:
        """
        # 文件保存路径通过输入的args.config中的project指定
        # 当 args.project 不存在时, 会通过Ultralytics Settings指定路径(第二选择), 参考ultralytics/docs/quickstart.md, 支持Modifying Settings
        # Ultralytics Settings 路径保存目前在/root/.config/Ultralytics/settings.yaml
        """
        # init
        cfg = yaml_load(args.config)
        if 'rtdetr' in os.path.basename(cfg['model']):
            model = CustomRTDETR(model_Bytes)
        elif 'yolo' in os.path.basename(cfg['model']):
            model = CustomYOLO(model_Bytes)
        else:
            raise NotImplementedError
        model_Bytes.close()
        # Display model information (optional)
        model.info()

        # cfg options
        settings.update({'runs_dir': './runs'})
        cfg_options = {**cfg_options, **{'exist_ok':True}}

        # 不使用cfg(相当于使用默认cfg)进行validate的结果和下方use_wrapper=False结果一致
        model.val(data=args.data, **cfg_options)

        # 使用cfg进行validate的结果和下方use_wrapper=False结果一致
        # cfg = {**cfg, **cfg_options}
        # cfg['data'] = args.data
        # model.val(**cfg)
    else:
        """
        # === deprecated ===
        # 不推荐使用, 需额外输入args.config进行初始化, 文件保存路径通过输入的args.config中的project指定
        # 当 args.project 不存在时, 会通过Ultralytics Settings指定路径(第二选择), 参考ultralytics/docs/quickstart.md, 支持Modifying Settings
        # Ultralytics Settings 路径保存目前在/root/.config/Ultralytics/settings.yaml
        # example: args.project is None: Results saved to /youtu/xlab-team4/ryanwfu/26_AutoModels/AutoProject/runs/detect/val8
        # Attention - onnx 格式的模型可以在下面的api中使用 
        """
        simple_cfg = dict(model=args.model, data=args.data, mode='val', exist_ok=True)
        cfg_options = {**simple_cfg, **cfg_options}
        cfg = get_cfg(args.config, overrides=cfg_options)
        validator = COCOYOLODetectionValidator(args=cfg)
        validator(model=args.model)


def val_format_only(args, model_loader: ModelLoader = None):

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    # Normal Import
    from ultralyticsp.models import CustomYOLO, CustomRTDETR
    from ultralyticsp.custom import yaml_load
    from ultralytics.cfg import get_cfg
    from ultralytics import settings
    from ultralytics.utils import colorstr

    # custom data cfg
    data_cfg = yaml_load(args.data)
    if isinstance(data_cfg['val'], list):
        coco_json_paths = [Path(os.path.join(data_cfg['path'], x)) for x in data_cfg['val']]
    else:
        coco_json_paths = [Path(os.path.join(data_cfg['path'], data_cfg['val']))]
    prefix = Path(data_cfg['val_prefix'])
    assert data_cfg['coco_format_json']
    roi_crop = data_cfg['roi_crop']
    roi_key = data_cfg['roi_key']

    if model_loader is None:
        model_loader = FileModelLoader(args.model)

    # init model
    cfg = yaml_load(args.config)
    if 'rtdetr' in os.path.basename(cfg['model']):
        model = CustomRTDETR(args.model)
    elif 'yolo' in os.path.basename(cfg['model']):
        model = CustomYOLO(args.model)
    else:
        raise NotImplementedError
    
    # process
    results = []
    for coco_json_path in coco_json_paths:
        
        # prepare data
        with coco_json_path.open('r') as f:
            content = json.load(f)
        images = content['images']
        annotations = content['annotations']

        imgid2annos = defaultdict(list)
        for anno_info in annotations:
            img_id = anno_info['image_id']
            imgid2annos[img_id].append(anno_info)

        catname2catid = {cat_info['name']: cat_info['id'] for cat_info in content['categories']}
        id2catname = model.names

        # inference
        for img_info in tqdm(images):
            img_id = img_info['id']

            img_path = prefix/img_info['file_name']

            # NOTE. modified by ryanwfu. 2023/10/10, compatible with Chinese paths in windows
            fb = str(img_path).encode('utf-8')
            img_data = cv2.imdecode(np.fromfile(fb, dtype=np.uint8), cv2.IMREAD_COLOR)
            # img_data = cv2.imread(str(img_path))    # BGR

            if roi_crop:
                if roi_key in img_info:
                    roi_info = img_info[roi_key]
                    rx1, ry1, rx2, ry2 = roi_info
                    img_data = img_data[ry1:ry2, rx1:rx2]
                else:
                    rx1, ry1 = 0, 0

            min_conf, max_det, custom_inference = 0.01, 100, True
            preds = model.predict(img_data, conf=min_conf, verbose=(not custom_inference), max_det=max_det, custom_inference=custom_inference)[0]

            if custom_inference:
                preds['boxes'] = preds['boxes'].cpu()
                xyxys = preds['boxes'][:, :4]
                confs = preds['boxes'][:, -2]
                labels = preds['boxes'][:, -1]
            else:
                preds = preds.cpu()
                xyxys = preds.boxes.xyxy
                confs = preds.boxes.conf
                labels = preds.boxes.cls

            for i in range(len(labels)):
                xmin, ymin, xmax, ymax = xyxys[i]
                w, h = xmax - xmin, ymax - ymin
                conf = float(confs[i])
                label = int(labels[i])
                if label < 0:
                    continue
                xywh = np.array([xmin, ymin, w, h])
                if roi_crop:
                    xywh += np.array([rx1, ry1, 0, 0])
                res_info = {'image_id': img_id, 'category_id': catname2catid[id2catname[label]], 
                            'bbox': xywh.tolist(), 'score': conf}
                results.append(res_info)

    # save
    save_dir = os.path.dirname(args.jsonfile_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prediction_results_save_path = args.jsonfile_prefix+'.bbox.json'
    with open(prediction_results_save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saving {prediction_results_save_path}...")
    print(f"Results saved to {colorstr('bold', os.path.dirname(prediction_results_save_path))}")


if __name__ == '__main__':
    args = parse_args()
    if not args.format_only:
        val(args)
    else:
        val_format_only(args)



