#!/bin/bash
BASE_DIR2=$(cd "$(dirname "$0")"; pwd)
CONFIG_PATH2=`python3 tools/pipeline/get_default_config.py ./init_setting.json config_path | awk 'END{print}'`
MODEL_DIR2=`python3 tools/pipeline/get_default_config.py ./init_setting.json model_dir | awk 'END{print}'`
RELEASE_DIR2=`python3 tools/pipeline/get_default_config.py ./init_setting.json release_dir | awk 'END{print}'`
echo "Using default config $CONFIG_PATH2"

# Model Path
IND_MODEL_PATH2=`python3 tools/pipeline/get_model_path.py $MODEL_DIR2 $CONFIG_PATH2 $RELEASE_DIR2 model_path`
RELEASE_FOLDER2=`python3 tools/pipeline/get_model_path.py $MODEL_DIR2 $CONFIG_PATH2 $RELEASE_DIR2 release_path`
export IND_MODEL_PATH2=$IND_MODEL_PATH2
export RELEASE_FOLDER2=$RELEASE_FOLDER2

# Python Path
BASE_DIR2=$(cd "$(dirname "$0")"; pwd)
export IND_PROJ_PATH2=$BASE_DIR2/../../
export PYTHONPATH=$IND_PROJ_PATH2
export PYTHONPATH=$IND_PROJ_PATH2/3rd_party:$PYTHONPATH

echo "Setting model path $IND_MODEL_PATH2"
echo "Setting project path $IND_PROJ_PATH2"

python3 tools/jitai_tools/auto_release.py
