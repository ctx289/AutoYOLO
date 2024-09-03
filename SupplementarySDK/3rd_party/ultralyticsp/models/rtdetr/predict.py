import torch
from ultralytics.utils import ops
from ultralyticsp.engine.predictor import CustomBasePredictor
from ultralytics.models.rtdetr.predict import RTDETRPredictor


class COCORTDETRPredictor(CustomBasePredictor, RTDETRPredictor):
    
    # NOTE. added by ryanwfu 2023/09/18
    def custom_postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions and returns a list of Results objects."""
        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            oh, ow = orig_img.shape[:2]
            if not isinstance(orig_imgs, torch.Tensor):
                pred[..., [0, 2]] *= ow
                pred[..., [1, 3]] *= oh
            results.append(dict(names=self.model.names, boxes=pred))
        return results