import re
import os
import argparse
import json, yaml
import tools as interface   # ultralytics-plus/tools


def yaml_load(file='data.yaml'):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        return data
    

def yaml_dump(cfg, save_path):
    # 将字典转换为 YAML 格式
    cfg_data = yaml.dump(cfg, sort_keys=False)

    # 将 YAML 数据保存到文件
    with open(save_path, 'w') as file:
        file.write(cfg_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-model-cfg", type=str)
    parser.add_argument("--template-data-cfg", type=str)
    parser.add_argument("--train-json", type=str)
    parser.add_argument("--val-json", type=str)
    parser.add_argument("--test-json", type=str)
    parser.add_argument("--output-model-cfg", type=str)
    parser.add_argument("--output-data-cfg", type=str)
    parser.add_argument("--autoresolution", action="store_true", help="whether use auto resolution")
    args = parser.parse_args()

    # data cfg
    # print(f'======> parse {args.template_data_cfg}')
    data_cfg = yaml_load(args.template_data_cfg)
    train_json = args.train_json.split(',')
    data_cfg['train'] = train_json if len(train_json) > 1 else train_json[0]
    val_json = args.val_json.split(',')
    data_cfg['val'] = val_json if len(val_json) > 1 else val_json[0]
    test_json = args.test_json.split(',')
    data_cfg['test'] = test_json if len(test_json) > 1 else test_json[0]
    # open train_json[0] for classes
    with open(train_json[0], 'r') as f:
        content = json.load(f)
    classes = dict()
    for idx, category in enumerate(content['categories']):
        classes[idx] = category['name']
    data_cfg['names'] = classes
    # parse crop info
    roi_crop, roi_key = False, 'crop_info'
    for image in content['images']:
        if 'outer' in image:
            roi_crop = True
            roi_key = 'outer'
            break
        elif 'crop_info' in image:
            roi_crop = True
            roi_key = 'crop_info'
            break
    data_cfg['roi_crop'] = roi_crop
    data_cfg['roi_key'] = roi_key

    # save data cfg
    if not os.path.exists(os.path.dirname(args.output_data_cfg)):
        os.makedirs(os.path.dirname(args.output_data_cfg))
    yaml_dump(data_cfg, args.output_data_cfg) 
    print(f'======> save data cfg : {args.output_data_cfg}')

    # model cfg
    # print(f'======> parse {args.template_model_cfg}')
    model_cfg = yaml_load(args.template_model_cfg)
    # imgsz
    if args.autoresolution:
        resolution = interface.AutoResolution(data_cfg['train'], enable_pose=False)
        imgsz, _ = resolution.get_recommended_resolution()
        model_cfg['imgsz'] = imgsz
        print(f'AutoResolution: imgsz={imgsz}')

    # save model cfg
    if not os.path.exists(os.path.dirname(args.output_model_cfg)):
        os.makedirs(os.path.dirname(args.output_model_cfg))
    yaml_dump(model_cfg, args.output_model_cfg)
    print(f'======> save model cfg : {args.output_model_cfg}')
    