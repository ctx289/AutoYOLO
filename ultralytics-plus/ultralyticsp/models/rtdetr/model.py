# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
RT-DETR model interface
"""
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from .train import COCORTDETRTrainer
from .val import COCORTDETRValidator
from .predict import COCORTDETRPredictor
from ultralyticsp.engine.model import CustomModel


class CustomRTDETR(CustomModel):
    """
    RTDETR model interface.
    """

    def __init__(self, model='rtdetr-l.pt') -> None:
        import _io
        if isinstance(model, _io.BytesIO) or isinstance(model, _io.BufferedReader):
            pass
        elif model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt file or *.yaml file or _io.BytesIO or  _io.BufferedReader for encrypt.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self):
        return {
            'detect': {
                'predictor': COCORTDETRPredictor,
                'validator': COCORTDETRValidator,
                'trainer': COCORTDETRTrainer,
                'model': RTDETRDetectionModel}}