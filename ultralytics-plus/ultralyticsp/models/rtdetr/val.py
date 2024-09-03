from pathlib import Path

import cv2
import numpy as np
import torch
import json
import os

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.models.rtdetr import RTDETRValidator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.checks import check_requirements

from ultralyticsp.data import COCOYOLODataset
from ultralyticsp.engine.validator import CustomBaseValidator
from ultralyticsp.data.augment import custom_v8_transforms

__all__ = 'COCORTDETRValidator',  # tuple or list

# TODO: Temporarily, RT-DETR does not need padding.
class COCORTDETRDataset(COCOYOLODataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    # NOTE: add stretch version load_image for rtdetr mosaic
    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image

                # NOTE. modified by ryanwfu. 2023/09/28, compatible with Chinese paths in windows
                fb = f.encode('utf-8')
                im = cv2.imdecode(np.fromfile(fb, dtype=np.uint8), cv2.IMREAD_COLOR)
                # im = cv2.imread(f)  # BGR

                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw

            # modified by ryanwfu
            if self.roi_crop:
                label_ = self.labels[i]
                roi_ = label_['roi_info']
                im = im[roi_[1]:roi_[3], roi_[0]:roi_[2]]
                h0, w0 = im.shape[:2]
            
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def build_transforms(self, hyp=None):
        """Temporarily, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # NOTE. modified by ryanwfu 2023/10/11. Support CopyPasteDet.
            transforms = custom_v8_transforms(self, self.imgsz, hyp, stretch=True)
            # transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class COCORTDETRValidator(CustomBaseValidator, RTDETRValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return COCORTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)
    
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
            predn[..., [0, 2]] *= shape[1] / self.args.imgsz  # native-space pred
            predn[..., [1, 3]] *= shape[0] / self.args.imgsz  # native-space pred

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(bbox)  # target boxes
                tbox[..., [0, 2]] *= shape[1]  # native-space pred
                tbox[..., [1, 3]] *= shape[0]  # native-space pred
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                # NOTE: To get correct metrics, the inputs of `_process_batch` should always be float32 type.
                correct_bboxes = self._process_batch(predn.float(), labelsn)
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
        LOGGER.info(f'\n escape instance Num: {escape_int_num}, overkill instance Num: {overkill_ins_num}\n' )
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
