## Ultralytics-plus scripts

### Some useful scripts

#### Visualize coco json file

- ***coco_json_visualization.py***
```
COCO_JSON_PATH=/youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/annotations/train.json
PREFIX='/'
SAVE_DIR='./vis/'

python3 ./scripts/coco_json_visualization.py --coco_json_path $COCO_JSON_PATH --prefix $PREFIX --save_dir $SAVE_DIR
```

#### Run inference on validation set in data config and visualize the results

- ***get_val_visualization.py***
```
MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default/train
SAVE_DIR=./runs/detect/val

python3 ./scripts/get_val_visualization.py --config $MODELCONFIG --model $WORKDIR/weights/best.pt --data $DATACONFIG --save_dir $SAVE_DIR
```

#### Calculate the overkill and missed rate through prediction json and coco json file

- ***img_level_validation.py***
```
COCO_JSON_PATH=/youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/annotations/val.json
PREDICTION_JSON_PATH=./runs/detect/val/test.bbox.json
PREFIX=/
SAVE_DIR=./runs/detect/val

python3 ./scripts/img_level_validation.py --coco_json_path $COCO_JSON_PATH --prediction_json_path $PREDICTION_JSON_PATH\
      --prefix $PREFIX --save_dir $SAVE_DIR --omit_realrecall
```
