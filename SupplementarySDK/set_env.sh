#!/bin/bash
BASE_DIR2=$(cd "$(dirname "$0")"; pwd)
CONFIG_PATH2=`python3 tools/pipeline/get_default_config.py ./init_setting.json config_path | awk 'END{print}'`
MODEL_DIR2=`python3 tools/pipeline/get_default_config.py ./init_setting.json model_dir | awk 'END{print}'`
RELEASE_DIR2=`python3 tools/pipeline/get_default_config.py ./init_setting.json release_dir | awk 'END{print}'`
echo "Using default config $CONFIG_PATH2"

# Python Path
export BASE_DIR2=$(cd "$(dirname "$0")"; pwd)
export IND_PROJ_PATH2=$BASE_DIR2
export PYTHONPATH=$IND_PROJ_PATH2:$PYTHONPATH
export PYTHONPATH=$IND_PROJ_PATH2/3rd_party:$PYTHONPATH
echo "Finish setting PYTHONPATH"

# Model Path
# IND_MODEL_PATH2=`python3 tools/pipeline/get_model_path.py $MODEL_DIR2 $CONFIG_PATH2 $RELEASE_DIR2 model_path`
IND_MODEL_PATH2=$IND_PROJ_PATH2/$MODEL_DIR2
export IND_MODEL_PATH2=$IND_MODEL_PATH2

# other sdk
# export PYTHONPATH=/youtu/xlab-team4/ryanwfu/24_second_sdk/industrial_eval_project:$PYTHONPATH
# export PYTHONPATH=$IND_PROJ_PATH2/pip:$PYTHONPATH
# export PYTHONPATH=$IND_PROJ_PATH2/pip_ultralytics:$PYTHONPATH
# export LD_LIBRARY_PATH=$IND_PROJ_PATH2/pip/paddle/libs/
# echo "Setting LD_LIBRARY_PATH $LD_LIBRARY_PATH"

echo "Setting model path $IND_MODEL_PATH2"
echo "Setting project path $IND_PROJ_PATH2"

if [ $# -eq 0 ]; then
    echo "Finish setting env"
    exec bash
else
    $@
fi