""" Dataset py for fuchi data
    custom anno loading code in here
"""
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class TongdaCocoDatasetLW(CocoDataset):

    def _convert_KL_to_LW(self, anno):
        """ convert category id of KL2 to LW1
        """
        if anno['category_id'] == 2:
            anno['category_id'] = 1

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            self._convert_KL_to_LW(ann)

            bbox = [x1, y1, x1 + w, y1 + h]

            # to ignore small ys class defect
            label = self.CLASSES[self.cat2label[ann['category_id']]]
            # if (label in ['YW', 'ZL', 'YS', 'QL']) and (ann['area'] < 3000):
            #     ann['iscrowd'] = 1

            # if ql zl and is ok
            if (label in ['LW', 'KL', 'QS', 'HS', 'YW', 'ZL', 'YS', 'QL', 'BX'] and ann.get('isOK', False)):
                ann['iscrowd'] = 1

            if ann.get("isKBJ", False):
                ann['iscrowd'] = 1

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)
        return ann

