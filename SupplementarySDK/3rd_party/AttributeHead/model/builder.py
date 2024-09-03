from mmcv.utils import Registry
from mmdet.models.builder import MODELS

ATTR_MODELS = Registry("models", scope="AttributeHead", parent=MODELS)

ATTR_HEADS = ATTR_MODELS
ATTR_BACKBONES = ATTR_MODELS
ATTR_NECKS = ATTR_MODELS
ATTR_LOSSES = ATTR_MODELS
ATTR_DETECTORS = ATTR_MODELS
