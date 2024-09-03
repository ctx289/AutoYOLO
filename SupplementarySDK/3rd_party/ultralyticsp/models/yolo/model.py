# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
YOLO model interface
"""
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel
from ultralyticsp.models.yolo.detect import COCOYOLODetectionTrainer, COCOYOLODetectionValidator, COCOYOLODetectionPredictor
from ultralyticsp.engine.model import CustomModel


class CustomYOLO(CustomModel):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            'detect': {
                'model': DetectionModel,
                'trainer': COCOYOLODetectionTrainer,
                'validator': COCOYOLODetectionValidator,
                'predictor': COCOYOLODetectionPredictor},
            }
