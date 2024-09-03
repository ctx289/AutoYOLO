## 数据
平台提供三个coco json 文件
- TRAIN_JSON
- VAL_JSON
- TEST_JSON

coco json 文件中， images 字段包含了 crop_info 字段，用于指示crop的区域

``` 
{
"images": [
	{
      "id": 2, 
      "width": 1280,
      "height": 1024,
      "file_name": "/youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/train/NG/S00184_C02_P002_L0_PI84_G1_M1_20230711032203.png",
      "sample": "YZ_BOTTLECAP1-train-NG-S00184_C02_P002_L0_PI84_G1_M1_20230711032203",
      "sample_cat": "YZPG1_ZW",
      "image_cats": [
        "YZPG1_ZW"
      ],
      "sample_cats": [
        "YZPG1_ZW"
      ],
      "flags": {},
      "crop_info": [
        220,
        200,
        966,
        702
      ]
	},
	...
	]
"annotations":[...],
"categories":[...],
}
```
## 训练
训练代码适配crop_info，实现crop子图训练
``` 
roi = img_info['crop_info']
im = im[roi[1]:roi[3], roi[0]:roi[2]]
box[:2] -= np.array(roi[:2], dtype=np.float64)
```
## 推理
推理不需要进行额外适配，平台调用sdk_entry.yaml中指定的interface.inference接口
``` 
module: algorithm_interface_sdk
interface: AlgorithmInterfacePredictor
```
如果是带数据预处理(Crop)的比赛，平台在调用interface.inference时会传入裁剪好的 image_data，得到推理结果后平台会映射到原图。

---

## Demo
https://git.woa.com/YoutuIndustrialAI/AutoModels/AutoUltralytics

##### Files that must be included

- docker.yaml
- exec.sh
- train.sh
- export.sh
- parser/*（解析config/yaml/ppl_config文件）
- 模型训练代码（例：ultralytics-plus、mmdetection-plus）
- SDK代码（例：SupplementarySDK）

平台依次调用Demo的 exec.sh、train.sh、export.sh

```python
# 生成训练config文件
sh exec.sh $WORK_DIR $TRAIN_JSON $VAL_JSON $TEST_JSON
# 开始训练
sh train.sh $WORK_DIR $TRAIN_JSON $VAL_JSON $TEST_JSON
# 导出SDK
sh export.sh $WORK_DIR $TRAIN_JSON $VAL_JSON $TEST_JSON
```
WORK_DIR、TRAIN_JSON、VAL_JSON、TEST_JSON 均由平台指定。
最后会在WORK_DIR处生成推理的sdk，平台运行推理的sdk得到最后的指标。

##### docker.yaml

``` 
GPUName: V100
host_num: 1
host_gpu_num: 1
image_full_name: mirrors.tencent.com/deep-xlab/torch:py38-torch1.12.0-detectron2-mmcv1.6.0-paddle2.4.2-v4.1
cuda_version: 11.0
```

