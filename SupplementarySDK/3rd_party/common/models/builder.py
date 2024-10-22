from mmcv.utils import Registry
from mmdet.models.builder import MODELS

COMMON_MODELS = Registry("models", scope="common", parent=MODELS)

COMMON_HEADS = COMMON_MODELS
COMMON_BACKBONES = COMMON_MODELS
COMMON_NECKS = COMMON_MODELS
COMMON_LOSSES = COMMON_MODELS
COMMON_DETECTORS = COMMON_MODELS
