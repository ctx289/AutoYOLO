# Ultralytics YOLO 🚀, AGPL-3.0 license

from .train import COCOYOLODetectionTrainer
from .val import COCOYOLODetectionValidator
from .predict import COCOYOLODetectionPredictor

__all__ = 'COCOYOLODetectionTrainer', 'COCOYOLODetectionValidator', 'COCOYOLODetectionPredictor'
