from mmcv.utils import Registry
from mmdet.datasets.builder import DATASETS, PIPELINES

ATTR_DATASET = Registry("dataset", parent=DATASETS, scope="AttributeHead")
ATTR_PIPELINES = Registry("pipeline", parent=PIPELINES, scope="AttributeHead")
