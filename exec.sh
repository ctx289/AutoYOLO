WORK_DIR=$1
TRAIN_JSON=$2
VAL_JSON=$3
TEST_JSON=$4

set -e

# 0. 设置环境变量
export PYTHONPATH=`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/ultralytics-plus

# 1. 生成训练config(yaml)文件
TEMPLATE_MODEL_CFG=./template/yolov8n_default.yaml
TEMPLATE_DATA_CFG=./template/data_cfg.yaml

OUTPUT_MODEL_CFG=$WORK_DIR/cfg/model_cfg.yaml
OUTPUT_DATA_CFG=$WORK_DIR/cfg/data_cfg.yaml

python3 ./parser/parse_train_cfg.py --template-model-cfg $TEMPLATE_MODEL_CFG --template-data-cfg $TEMPLATE_DATA_CFG\
    --output-model-cfg $OUTPUT_MODEL_CFG --output-data-cfg $OUTPUT_DATA_CFG\
    --train-json $TRAIN_JSON --val-json $VAL_JSON --test-json $TEST_JSON 
# --autoresolution


# 2. 生成模板图的json文件（deprecated）   
# ROIDIR=$WORK_DIR/roi_dir/
# python3 ./parser/parse_roi_data.py --train-json $TRAIN_JSON --save-path $ROIDIR

# 3. 生成threshold.yaml文件
OUTPUT_THRESHOLD_YAML=$WORK_DIR/cfg/threshold.yaml
python3 ./parser/parse_threshold_yaml.py --train-json $TRAIN_JSON --output-threshold-yaml $OUTPUT_THRESHOLD_YAML --default-threshold 0.1

# 4. 指定 MODELCONFIG 和 DATACONFIG 到环境变量 (deprecated)
# export MODELCONFIG=$OUTPUT_MODEL_CFG
# export DATACONFIG=$OUTPUT_DATA_CFG
# export THRESHOLDYAML=$OUTPUT_THRESHOLD_YAML

echo "Finish exec.sh."

# shell进程替换为子shell进程, 子shell进程中的环境变量生效(与bash/sh命令相配合，如果是source执行的命令则可以不用)
# exec bash
