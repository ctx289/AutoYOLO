import json
import sys

from mmcv import Config

if __name__ == "__main__":
    default_json_path = sys.argv[1]
    mode = sys.argv[2]
    json_data = Config.fromfile(default_json_path)

    if mode == "config_path":
        if not json_data['CONFIG_PATH']:
            print("EmptyConfigInInitSetting.py")
        else:
            print(json_data['CONFIG_PATH'])
    elif mode == "model_dir":
        if json_data['RELEASE_DIR'] == "":
            print("./")
        else:
            print(json_data['MODEL_DIR'])
    elif mode == "release_dir":
        if json_data['RELEASE_DIR'] == "":
            print("./")
        else:
            print(json_data['RELEASE_DIR'])
