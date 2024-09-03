## `DataAOI Platform Train Demo - Quick Start`

### Files that must be included

- docker.yaml
- exec.sh
- train.sh
- export.sh
- parser/*（解析config/yaml/ppl_config文件）
- 模型训练代码（例：ultralytics-plus、mmdetection-plus）
- SDK代码（例：SupplementarySDK）

### Git clone steps

```bash
git clone https://git.woa.com/YoutuIndustrialAI/AutoModels/AutoUltralytics.git
cd AutoUltralytics
# for submodule
git submodule init
git submodule update
```

### Run steps

- ***生成训练config文件***

```
WORK_DIR=./work_dir
TRAIN_JSON=/annotations/train.json
VAL_JSON=/annotations/val.json
TEST_JSON=/annotations/val.json

sh ./exec.sh $WORK_DIR $TRAIN_JSON $VAL_JSON $TEST_JSON
```
ps: source命令在当前shell环境中执行脚本，而bash/sh命令在新的shell环境中执行脚本

- ***开始训练***

```
WORK_DIR=./work_dir
TRAIN_JSON=/annotations/train.json
VAL_JSON=/annotations/val.json
TEST_JSON=/annotations/val.json

sh train.sh $WORK_DIR $TRAIN_JSON $VAL_JSON $TEST_JSON
```

- ***导出SDK***

```
WORK_DIR=./work_dir
TRAIN_JSON=/annotations/train.json
VAL_JSON=/annotations/val.json
TEST_JSON=/annotations/val.json

sh export.sh $WORK_DIR $TRAIN_JSON $VAL_JSON $TEST_JSON
```

### 多json输入

- ***支持多个json作为输入，以英文,隔开***

```
WORK_DIR=./work_dir
TRAIN_JSON=/annotations/train.json
VAL_JSON=/annotations/val.json
TEST_JSON=/annotations/val.json
```

### 模型训练配置

- ***支持多种yolov8模型，可修改模型训练配置文件./template/yolov8n_default.yaml***
- ***也可以在train.sh中提供训练配置参数***
```
python3 ./ultralytics-plus/tools/train.py $MODELCONFIG --data $DATACONFIG --work-dir $DET_WORKDIR --cfg-options batch=16 imgsz=100 epochs=100
```
