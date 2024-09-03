WORK_DIR=$1
TRAIN_JSON=$2
VAL_JSON=$3
TEST_JSON=$4

set -e

# 1. 设置环境变量
MODELCONFIG=$WORK_DIR/cfg/model_cfg.yaml
DATACONFIG=$WORK_DIR/cfg/data_cfg.yaml

# 2. 执行模型训练
export PYTHONPATH=`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/ultralytics-plus
DET_WORKDIR=$WORK_DIR/train
python3 ./ultralytics-plus/tools/train.py $MODELCONFIG --data $DATACONFIG --work-dir $DET_WORKDIR

echo "Finish train.sh."
