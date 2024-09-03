import argparse
import json
import re
import os
import yaml
from mmcv import Config


def yaml_dump(cfg, save_path):
    # 将字典转换为 YAML 格式
    cfg_data = yaml.dump(cfg)

    # 将 YAML 数据保存到文件
    with open(save_path, 'w') as file:
        file.write(cfg_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=str)
    parser.add_argument("--default-threshold", type=float, default=0.1)
    parser.add_argument("--output-threshold-yaml", type=str)
    args = parser.parse_args()

    train_json = args.train_json.split(',')
    with open(train_json[0], 'r') as f:
        content = json.load(f)
    threshold = dict()
    for idx, category in enumerate(content['categories']):
        threshold[category['name']] = args.default_threshold
    
    # save
    yaml_dump(threshold, args.output_threshold_yaml)
    print(f'======> save threshold yaml : {args.output_threshold_yaml}')
    
    