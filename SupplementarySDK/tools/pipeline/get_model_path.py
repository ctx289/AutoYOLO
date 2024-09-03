""" get model path for envir variables
"""
import os
import sys


def get_model_path(model_dir, config_path):
    config_name = os.path.basename(config_path)

    if config_name.endswith(".json"):
        config_dir = config_name.split(".json")[0]
    elif config_name.endswith(".py"):
        config_dir = config_name.split(".py")[0]
    else:
        raise ValueError(
            "Unsupported config format, expected .py or .json but got {}".
            format(config_name))

    model_path = os.path.join(model_dir, config_dir)
    return model_path


if __name__ == "__main__":
    model_dir = sys.argv[1]
    config_path = sys.argv[2]
    release_dir = sys.argv[3]
    mode = sys.argv[4]
    release_folder = get_model_path(release_dir, config_path)
    model_path = os.path.join(release_folder, model_dir)

    if mode == "model_path":
        print(model_path, end="")
    elif mode == "release_path":
        print(release_folder, end="")
    else:
        raise ValueError("Unsupport mode {}".format(mode))
