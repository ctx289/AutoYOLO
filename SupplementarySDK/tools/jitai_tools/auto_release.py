""" Auto release for project
"""

import argparse
import json
import os
import shutil
import sys
from copy import copy
from distutils.dir_util import copy_tree

from indproj2.algos.utils.config_utils import recursive_change_keys
from mmcv import Config
from utils2.config_utils import fix_dict, get_default_config_path
from utils2.path_utils import check_exist_create


def change_init_setting_json(release_dir, proj_dir):
    from_init_setting_path = os.path.join(proj_dir, "init_setting.json")
    with open(from_init_setting_path, "r") as f:
        json_data = json.load(f)
    json_data['CONFIG_PATH'] = json_data['CONFIG_PATH'].replace(
        ".py", "_release.py")
    init_setting_path = os.path.join(release_dir, "init_setting.json")
    with open(init_setting_path, "w") as f:
        json.dump(json_data, f)
    return init_setting_path


def change_init_setting_json_autoproject(release_dir, proj_dir, config_path):
    from_init_setting_path = os.path.join(proj_dir, "init_setting.json")
    with open(from_init_setting_path, "r") as f:
        json_data = json.load(f)
    json_data['CONFIG_PATH'] = config_path.replace(".py", "_release.py")
    init_setting_path = os.path.join(release_dir, "init_setting.json")
    with open(init_setting_path, "w") as f:
        json.dump(json_data, f)
    return init_setting_path


def copy_packing(config, model_dir):
    # normal packing
    file_folder_name = os.path.basename(os.path.dirname(config))
    save_folder = os.path.join(model_dir, file_folder_name)
    check_exist_create(save_folder)
    print("Copying", config, "to", save_folder)
    if os.path.isdir(config):
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        shutil.copytree(config, save_folder)
    else:
        shutil.copy(config, save_folder)


def move_data_to_model_path(key, config, model_dir, encrypt=False):
    """ move_data_to_model_path, auto packing script
        This function will recursively check each key in the config
            and run following logic
        1. if key is config/ckpt, will change path to relative path
            and pack to model_dir. The path format will keep depth=2.
        2. if key starts with roi(roi_dir)/high_roi/calib(calib_dir)/template_root/
            run logic as 1
        3. if value is a folder, run logic as 1

    Args:
        key (str): key value in the config, each key will be recursively checked
        config (dict/list/str): the upper level of key, used as config[key]
        model_dir (str): packing directory

    Returns:
        config: with value changed to relative path
    """
    key_value = config[key] if key is not None else config
    if isinstance(key_value, (dict, Config)):
        encrypt_model = key_value.get("encrypt", encrypt)
        for key, value in key_value.items():
            key_value[key] = move_data_to_model_path(key, key_value, model_dir,
                                                     encrypt_model)
    elif isinstance(key_value, list):
        for idx in range(len(key_value)):
            key_value[idx] = move_data_to_model_path(idx, key_value, model_dir)
    elif isinstance(key_value, str):
        if key in ["ckpt", "config"]:
            copy_packing(key_value, model_dir)
            file_name = os.path.basename(key_value)
            file_folder_name = os.path.basename(os.path.dirname(key_value))
            key_value = os.path.join("$IND_MODEL_PATH2", file_folder_name,
                                     file_name)
        elif isinstance(key, str):
            if key.startswith("roi") \
                    or key.startswith("high_roi") \
                    or key.startswith("calib") \
                    or key == "template_root" \
                    or os.path.isdir(key_value):
                folder_name = os.path.basename(key_value.rstrip("/"))
                check_exist_create(model_dir)
                print("Copying", key_value, "to",
                        os.path.join(model_dir, folder_name))
                copy_tree(key_value, os.path.join(model_dir, folder_name))
                key_value = os.path.join("$IND_MODEL_PATH2", folder_name)

    return key_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoproject',
                        action='store_true',
                        help="config path if not specified will use default")
    parser.add_argument('--config-path',
                        type=str,
                        default=None,
                        required=False,
                        help="config path if not specified will use default")
    args = parser.parse_args()
    if args.autoproject:
        assert args.config_path is not None
        config_path = args.config_path
    else:
        config_path = get_default_config_path()

    config = Config.fromfile(config_path)
    config = fix_dict(config)

    model_dir = os.getenv("IND_MODEL_PATH2")
    release_dir = os.getenv("RELEASE_FOLDER2")
    proj_dir = os.getenv("IND_PROJ_PATH2")
    release_dir = os.path.join(os.path.dirname(release_dir),
                               os.path.basename(release_dir) + "_release")
    model_dir = os.path.join(release_dir, os.path.basename(model_dir))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # add change items for release
    if config.get("RELEASE_CHANGE_ITEMS", None) is not None:
        recursive_change_keys(config.pipeline, config.RELEASE_CHANGE_ITEMS)
        config.pop("RELEASE_CHANGE_ITEMS")

    config = move_data_to_model_path(None, config, model_dir)

    config.dump(config_path.replace(".py", "_release.py"))
    print("Config saved to {}".format(config_path.replace(".py", "_release.py")))
    print("Models saved to {}".format(model_dir))

    # copy code
    print("Copying code from {} to {}".format(proj_dir, release_dir))
    copy_tree(proj_dir, release_dir)

    if args.autoproject:
        config_name = os.path.basename(config_path)
        relative_config_path = os.path.join('./configs/autoproject/',config_name)
        config_path2 = os.path.join(release_dir, relative_config_path)
        config.dump(config_path2.replace(".py", "_release.py"))
        print("Config saved to {}".format(config_path2.replace(".py", "_release.py")))

    white_list = ["rpfs", ".git"]
    for w_folder in white_list:
        w_path = os.path.join(release_dir, w_folder)
        if os.path.exists(w_path):
            if os.path.isdir(w_path):
                shutil.rmtree(w_path)
            else:
                os.remove(w_path)
    
    if args.autoproject:
        changed_init_setting_path = change_init_setting_json_autoproject(release_dir, proj_dir, relative_config_path)
    else:
        changed_init_setting_path = change_init_setting_json(release_dir, proj_dir)
    print("Changed config in init_setting.json to ", changed_init_setting_path)
