import os
from pathlib import Path
import json

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops

from ultralytics.models.yolo.detect import DetectionValidator
from ultralyticsp.data import build_coco_yolo_dataset
from ultralyticsp.engine.validator import CustomBaseValidator

class COCOYOLODetectionValidator(CustomBaseValidator, DetectionValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_coco_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)
    
    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')  # is COCO
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []
        
        # NOTE. 2023/09/05 added by ryanwfu
        val = val[0] if isinstance(val, list) else val
        with open(val, 'r') as f:
            content = json.load(f)
        self.catname2catid = {cat_info['name']: cat_info['id'] for cat_info in content['categories']}
    
    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                # NOTE. 2023/09/05 modified by ryanwfu
                self.pred_to_json(predn, batch['im_file'][si], batch['roi_info'][si], batch['image_id'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)
    
    def pred_to_json(self, predn, filename, roi_info, image_id):
        """Serialize YOLO predictions to COCO json format."""

        # NOTE. 2023/09/05 modified by ryanwfu
        # stem = Path(filename).stem
        # image_id = int(stem) if stem.isnumeric() else stem

        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

        # NOTE. 2023/09/05 modified by ryanwfu
        if roi_info is not None:
            box = box.cpu()
            rx1, ry1, rx2, ry2 = roi_info
            box[:, :2] += np.array([rx1, ry1])

        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                # NOTE. 2023/09/05 modified by ryanwfu
                'category_id': self.catname2catid[self.names[int(p[5])]],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})
    
    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        # 230918 joefzhou: print the instant-level metric from confusion matrix.
        pred_json = self.save_dir / 'predictions.json'  # predictions
        c_matrix = self.confusion_matrix.matrix
        escape_int_num = sum(c_matrix[-1, :])
        overkill_ins_num = sum(c_matrix[:, -1])

        # NOTE. (ryanwfu) c_matrix is calculated using a fixed threshold of 0.25, equivalent to predict; 2023/10/09
        LOGGER.info(f'\n when conf equals 0.25, escape instance Num: {escape_int_num}, overkill instance Num: {overkill_ins_num}\n' )
        
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats
