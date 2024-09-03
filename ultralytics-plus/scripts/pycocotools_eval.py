import json
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


coco_json_file = '/youtu/xlab-team4/share/datasets/YZ_BOTTLECAP1/annotations/val.json'
result_file = '/youtu/xlab-team4/ryanwfu/28_ultralytics/ultralytics-plus/runs/detect/val/test.bbox.json'
# result_file = '/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default_add_badcase/train/predictions.json'

# 加载 COCO JSON 文件
coco_gt = COCO(coco_json_file)

# 加载生成的 test.bbox.json / predictions.json 文件
try:
    coco_dt = coco_gt.loadRes(result_file)
except:
    with open(result_file, 'r') as f:
        content = json.load(f)
    if isinstance(content, list) and len(content) == 0:
        print('\nThe testing results is empty.\n')
        sys.exit()
    else:
        print('\nThe testing results error.\n',)
        raise Exception('testing results error')

# 创建评测对象
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# 运行评测
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()