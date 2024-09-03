import os

from mmcv import Config


def get_default_config_path():
    ind_proj_path = os.getenv("IND_PROJ_PATH2")
    init_setting_path = os.path.join(ind_proj_path, "init_setting.json")

    init_setting = Config.fromfile(init_setting_path)
    config_path = os.path.join(ind_proj_path, init_setting['CONFIG_PATH'])
    return config_path


def fix_value(val):
    if isinstance(val, str):
        val = val.replace("$IND_MODEL_PATH2", os.path.abspath(os.getenv("IND_MODEL_PATH2")))
        val = val.replace("$IND_PROJ_PATH2", os.path.abspath(os.getenv("IND_PROJ_PATH2")))
    return val


def fix_dict(config):
    if isinstance(config, (list)):
        for i in range(len(config)):
            config[i] = fix_dict(config[i])
    elif isinstance(config, (Config, dict)):
        for k in config.keys():
            config[k] = fix_dict(config[k])
    elif isinstance(config, str):
        config = fix_value(config)

    return config
