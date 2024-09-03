import torch
from ultralytics.utils import ops
from ultralyticsp.engine.predictor import CustomBasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class COCOYOLODetectionPredictor(CustomBasePredictor, DetectionPredictor):
    
    # NOTE. added by ryanwfu 2023/09/18
    def custom_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(dict(names=self.model.names, boxes=pred))
        return results