WORK_DIR=$1
TRAIN_JSON=$2
VAL_JSON=$3
TEST_JSON=$4

set -e

# 0. 设置环境变量
THRESHOLDYAML=$WORK_DIR/cfg/threshold.yaml

# 1. parse ppl cfg
DET_WORKDIR=$WORK_DIR/train
template_pipeline_config_path=./template/autoproject_pipeline_v1.py
output_pipeline_config_path=./SupplementarySDK/configs/autoproject/autoproject_pipeline_v1_out.py

python3 ./parser/parse_pipeline_cfg.py --input_pipeline_config_path $template_pipeline_config_path\
 --output_pipeline_config_path $output_pipeline_config_path\
 --det_arch 'UltralyticsFasterDetector' --det_work_dir $DET_WORKDIR
echo "Finish parsing pipeline cfg."
echo ""

# 2. export sdk
#!/bin/bash
CONFIG_PATH2=$output_pipeline_config_path
MODEL_DIR2='./model_dir'
RELEASE_DIR2=$WORK_DIR
# Model Path
IND_MODEL_PATH2=`python3 ./SupplementarySDK/tools/pipeline/get_model_path.py $MODEL_DIR2 $CONFIG_PATH2 $RELEASE_DIR2 model_path`
RELEASE_FOLDER2=`python3 ./SupplementarySDK/tools/pipeline/get_model_path.py $MODEL_DIR2 $CONFIG_PATH2 $RELEASE_DIR2 release_path`
export IND_MODEL_PATH2=$IND_MODEL_PATH2
export IND_PROJ_PATH2=`pwd`/SupplementarySDK
export RELEASE_FOLDER2=$RELEASE_FOLDER2
# Python Path
export PYTHONPATH=$PYTHONPATH:`pwd`/SupplementarySDK
export PYTHONPATH=$PYTHONPATH:`pwd`/SupplementarySDK/3rd_party
# echo
echo "Setting project path $IND_PROJ_PATH2"
echo "Setting release dir $RELEASE_FOLDER2"
echo "Setting model path $IND_MODEL_PATH2"
# run
python3 ./SupplementarySDK/tools/jitai_tools/auto_release.py --autoproject --config-path $CONFIG_PATH2
echo ""

# 3. delete output_pipeline_config_path
output_pipeline_config_name=$(basename "$output_pipeline_config_path")
rm $output_pipeline_config_path
rm "$(dirname "$output_pipeline_config_path")/${output_pipeline_config_name%.*}""_release.py"
echo "Deleting temporary ppl config."

# 4. copy threshold.yaml to $RELEASE_FOLDER2
cp $THRESHOLDYAML $RELEASE_FOLDER2'_release'
echo "Copying threshold.yaml to sdk"

# 5. copy docker.yaml to $RELEASE_FOLDER2
cp ./docker.yaml $RELEASE_FOLDER2'_release'
echo "Copying docker.yaml to sdk"
echo ""

echo "Finish export.sh."


