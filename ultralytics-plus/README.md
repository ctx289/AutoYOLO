## `YouTu Industrial Ultralytics - Quick Start`

### Get Started

Docker for V100
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --network=host -v /apdcephfs:/apdcephfs -v /youtu/:/youtu/ -w `pwd` --shm-size 20G mirrors.tencent.com/deep-xlab/torch:py38-torch1.12.0-detectron2-mmcv1.6.0-paddle2.4.2-v4 /bin/bash
```

### Installation
Please refer to [ultralytics](https://pypi.org/project/ultralytics/) ([ultralytics 8.0.156](https://pypi.org/project/ultralytics/8.0.156/) is recommended).
```bash
# install from pip (recommended)
pip3 install ultralytics==8.0.156

# install from source
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -v -e .
```

### windows remote machine
```
10.8.187.36:36000  sh地下机房 2卡A4000（windows）

mac 远程 windows 桌面:

1. iOA“通道模式”切换为“PAC代理”
2. 下载安装 Microsoft Remote Desktop: https://install.appcenter.ms/orgs/rdmacios-k2vy/apps/microsoft-remote-desktop-for-mac/distribution_groups/all-users-of-microsoft-remote-desktop-for-mac
3. 使用远程桌面帐号登陆

Administrator, Admin@2023
```

### Train demo
- ***train on default single-gpu `[linux-default-gpu]`***
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`

MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

python3 ./tools/train.py $MODELCONFIG --data $DATACONFIG --work-dir $WORKDIR --cfg-options batch=16 imgsz=800
```

- ***train on specified single-gpu `[linux-specified-gpu]`***
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`

MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

python3 ./tools/train.py $MODELCONFIG --data $DATACONFIG --work-dir $WORKDIR --cfg-options batch=16 imgsz=800 --cuda_visible_devices 1
```

- ***train on single-gpu `[windows]`***
```bash
set PYTHONPATH=D:\ryanwfu\ultralytics-plus;%PYTHONPATH%

set MODELCONFIG=D:\ryanwfu\ultralytics-plus/ultralyticsp/cfg/platform/yolov8n_default.yaml
set DATACONFIG=D:\ryanwfu\ultralytics-plus/ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop_win.yaml
set WORKDIR=D:\ryanwfu\work-dir/yolov8n_default_yz_bottlecap1/train

python ./tools/train.py %MODELCONFIG% --data %DATACONFIG% --work-dir %WORKDIR% --cfg-options batch=16 imgsz=800
```

- ***train on multi-gpu devcloud***
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`

MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

# --batch is the total batch-size. It will be divided evenly to each GPU.
# --cuda_visible_devices actually specifies which gpus to use for training
python3 ./tools/train.py $MODELCONFIG --data $DATACONFIG --work-dir $WORKDIR --cfg-options batch=24 imgsz=800 device=0,1 --cuda_visible_devices 1,2

# Another way to run multi-gpu training, without using custom_generate_ddp_command
python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29500 ./tools/train.py $MODELCONFIG --data $DATACONFIG --work-dir $WORKDIR --cfg-options batch=24 imgsz=800 device=0,1 --cuda_visible_devices 0,1

# Engineering team calls the exe, equivalent to the previous command, 
python3 ./tools/run.py --nproc_per_node 2 --master_port 29500 --no_python --with_exe --master_addr localhost ./tools/train.exe $MODELCONFIG --data $DATACONFIG --work-dir $WORKDIR --cfg-options batch=24 imgsz=800 device=0,1 --cuda_visible_devices 0,1
```

### Test demo
- ***Test AP & Format `(recommend)`***
```bash
MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

python3 ./tools/val.py --config $MODELCONFIG --model $WORKDIR/weights/best.pt --data $DATACONFIG --work-dir $WORKDIR --cfg-options save_json=True --cuda_visible_devices 0
```
- ***Format Only & speed test***
```bash
MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
DATACONFIG=./ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

python3 ./tools/val.py --config $MODELCONFIG --model $WORKDIR/weights/best.pt --data $DATACONFIG --jsonfile-prefix $WORKDIR/test --format-only --cuda_visible_devices 0
```

### Predict demo
```bash
# Only here can you specify the device to use the gpu
MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
SOURCE=./assets/S00265_C02_P003_L0_PI84_G1_M1_20230711032203.png
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

python3 ./tools/predict.py --config $MODELCONFIG --model $WORKDIR/weights/best.pt --source $SOURCE --cfg-options conf=0.1
```

### Export demo
```bash
MODELCONFIG=./ultralyticsp/cfg/platform/yolov8n_default.yaml
WORKDIR=/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_yz_bottlecap1/train

python3 ./tools/export.py --config $MODELCONFIG --model $WORKDIR/weights/best.pt --format onnx --cfg-options opset=14
```

### AutoResolution
```bash
from tools import AutoResolution

coco_json_path = /youtu/xlab-team4/share/datasets/Defects/Apple2023_Cylinder/annotations/Cylinder_train_annotations.coco.json
AutoRES = AutoResolution(coco_json_path, enable_pose=False)
recommend_resolution, _ = AutoRES.get_recommended_resolution()
```

### For Pipeline

For putting into pipeline, we need to define four rules: model_init, model_inference, input and output

- **Official Usage with [Pipeline](https://git.woa.com/YoutuIndustrialAI/Projects/IndustrialProject)**

1. Init and Inference
```bash
from ultralyticsp.models import CustomYOLO
# init
model = CustomYOLO(ckpt)
# inference, if conf not specified, default is 0.25
output = model.predict([img], conf=0.01, verbose=False, max_det=100, custom_inference=False, device=self.device_id)[0]
```

2. Input is a single image path or a single image with type of numpy.ndarray/torch.Tensor in BGR channels, refers to [here](https://docs.ultralytics.com/modes/predict/#inference-sources)

3. Output refers to [here](https://git.woa.com/YoutuIndustrialAI/tiaoi-algos/ultralytics-plus/blob/master/tools/val.py#L172)

4. ckpt supports pt(yolov8n.pt), yaml(yolov8n.yaml) and BinaryIO(model_Bytes).


### Supported methods

- [yolov8n](https://docs.ultralytics.com/models/yolov8/)
- [rtdetr-l](https://docs.ultralytics.com/models/rtdetr/)

### Performance

- ***YZBOTTLECAP1***

| 模型cfg | 总实例数 | 漏检实例数 | 过杀实例数 | 总图片数 | NG图片数 | OK图片数 | 漏检图片数 | 过杀图片数 | Inference time(ms/im) |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| [yolov8n_default.yaml](ultralyticsp/cfg/platform/yolov8n_default.yaml) | 65 | `11` | `1` | 84 | 48 | 36 | `7` | `1` | 13 |
| [rtdetrl_default.yaml](ultralyticsp/cfg/platform/rtdetrl_default.yaml) | 65 | `15` | `5` | 84 | 48 | 36 | `3` | `0` | `99` |

* Set a fixed threshold to 0.2
* Inference time(ms/im) is tested on imgsz=800 and T4 GPU
* The inference time of rtdetrl is very different from the official published FPS=114

- ***YZBOTTLECAP2***

| 模型cfg | 总实例数 | 漏检实例数 | 过杀实例数 | 总图片数 | NG图片数 | OK图片数 | 漏检图片数 | 过杀图片数 | Inference time(ms/im) |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| [yolov8n_default.yaml](ultralyticsp/cfg/platform/yolov8n_default.yaml) | 21 | `6` | `5` | 47 | 20 | 27 | `4` | `1` | 13 |

* Set a fixed threshold to 0.2

### badcase iterative experiment

- ***YZBOTTLECAP1-yolov8n_default.yaml***

| 训练序号 | 总实例数 | 漏检实例数 | 过杀实例数 | 总图片数 | NG图片数 | OK图片数 | 漏检图片数 | 过杀图片数 | Inference time(ms/im) |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| yolov8n_1th | 65 | `11` | `1` | 84 | 48 | 36 | `7` | `1` | 13 |
| yolov8n_2th | 65 | `7` | `4` | 84 | 48 | 36 | `4` | `0` | 13 |

* Iterated yolov8n's `eight` image-level badcases during the first training (7 missed, 1 overkill)
* The `four` image-level badcases still missed in the second training are a subset of the first training
