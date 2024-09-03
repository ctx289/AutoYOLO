# Copyright (c) OpenMMLab. All rights reserved.
import logging

import mmcv
import numpy as np
import prettytable
from mmcv.utils import Registry, print_log
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from prettytable import PrettyTable
from terminaltables import AsciiTable

from .builder import ATTR_DATASET


@ATTR_DATASET.register_module()
class AttrCocoDataset(CocoDataset):

    ATTRIBUTES = (
        "person",
        "vehicle",
        "outdoor",
        "animal",
        "accessory",
        "sports",
        "kitchen",
        "food",
        "furniture",
        "electronic",
        "appliance",
        "indoor",
    )

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        attributes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
    ):
        super().__init__(
            ann_file,
            pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )
        if attributes is not None:
            if isinstance(attributes, tuple):
                attributes = list(attributes)
            assert isinstance(attributes, list)
            self.ATTRIBUTES = attributes

    # get the attribute label (edit)
    def ann2attrid(self, ann):
        if not hasattr(self, "catid2attr"):
            self.catid2attr = dict()
            for _, item in self.coco.cats.items():
                self.catid2attr[item["id"]] = item["supercategory"]

        catid = ann["category_id"]
        super_cat = self.catid2attr[catid]
        return self.ATTRIBUTES.index(super_cat)

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_attrs = []

        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            if ann.get("isKBJ", False):
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))
                gt_attrs.append(self.ann2attrid(ann))

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

        seg_map = img_info["filename"].replace("jpg", "png")

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            attrs=gt_attrs,
        )
        return ann

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                # has attrs
                if isinstance(result[label], list):
                    bboxes = result[label][0]
                    attrs = result[label][1]
                else:
                    bboxes = result[label]
                    attrs = None

                for i in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(bboxes[i][4])
                    data["category_id"] = self.cat_ids[label]

                    if attrs is not None:
                        if len(attrs[i]) > len(self.ATTRIBUTES):
                            data["attribute_id"] = np.argmax(attrs[i][:-1])
                        else:
                            data["attribute_id"] = np.argmax(attrs[i])
                        data["attribute_score"] = attrs[i][data["attribute_id"]]
                        data["attribute_name"] = self.ATTRIBUTES[data["attribute_id"]]

                    json_results.append(data)
        return json_results

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None,
    ):
        metrics = metric if isinstance(metric, list) else [metric]
        if "attr_acc" not in metric:
            return super(AttrCocoDataset, self).evaluate(
                results=results,
                metric=metrics,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thrs,
                metric_items=metric_items,
            )
        else:
            metrics.remove("attr_acc")
            eval_results = super(AttrCocoDataset, self).evaluate(
                results=results,
                metric=metrics,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thrs,
                metric_items=metric_items,
            )

            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if iou_thrs is None:
                iou_thrs = np.linspace(
                    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
                )

            cocoGt = self.coco

            msg = f"Evaluating attributes accuaracy"
            if logger is None:
                msg = "\n" + msg
            print_log(msg, logger=logger)

            metric = "bbox"
            iou_type = "bbox"
            if metric not in result_files:
                raise KeyError(f"{metric} is not in results")
            try:
                predictions = mmcv.load(result_files[metric])
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    "The testing results of the whole dataset is empty.",
                    logger=logger,
                    level=logging.ERROR,
                )
                return eval_results

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            num_attr = len(self.ATTRIBUTES)
            num_class = len(self.CLASSES)

            atrribute_results = np.zeros(
                (num_class, num_attr, num_attr + 1), dtype=np.long
            )
            for (picture_idx, class_id), ious in cocoEval.ious.items():
                if ious == []:
                    continue

                pred_index, gt_index = np.where(ious > 0.5)
                for gt_id, pred_id in zip(gt_index, pred_index):
                    gt = cocoEval._gts[(picture_idx, class_id)][gt_id]
                    pred = cocoEval._dts[(picture_idx, class_id)][pred_id]

                    pred_attr_id = pred["attribute_id"]
                    gt_attr_id = self.ann2attrid(gt)
                    if gt_attr_id == -1:
                        gt_attr_id = num_attr
                    class_idx = self.coco.getCatIds().index(class_id)
                    atrribute_results[class_idx][pred_attr_id][gt_attr_id] += 1

            if classwise:
                for class_idx, classid in enumerate(self.coco.get_cat_ids()):
                    nm = self.coco.loadCats(classid)[0]["name"]
                    show_table = PrettyTable()
                    show_table.field_names = (
                        ["pred | gt"]
                        + list(self.ATTRIBUTES)
                        + ["UNKNOW"]
                        + ["precision"]
                    )
                    for attr_idx, attr_nm in enumerate(self.ATTRIBUTES):
                        recall = atrribute_results[class_idx][attr_idx][attr_idx] / sum(
                            atrribute_results[class_idx][attr_idx]
                        )
                        show_table.add_row(
                            [attr_nm]
                            + atrribute_results[class_idx][attr_idx].tolist()
                            + [recall]
                        )
                    show_table.add_row(
                        ["recall"]
                        + [
                            atrribute_results[class_idx][attr_idx, attr_idx]
                            / sum(atrribute_results[class_idx][:, attr_idx])
                            for attr_idx in range(num_attr)
                        ]
                        + ["None"]
                        + ["None"]
                    )
                    print_log(msg="{}".format(nm), logger=logger)
                    print_log(msg=show_table, logger=logger)
            else:
                show_table = PrettyTable()
                show_table.field_names = (
                    ["pred | gt"] + list(self.ATTRIBUTES) + ["UNKNOW"] + ["recall"]
                )
                for attr_idx, attr_nm in enumerate(self.ATTRIBUTES):
                    recall = "{:0.3f}".format(
                        np.sum(atrribute_results[:, attr_idx, attr_idx])
                        / np.sum(atrribute_results[:, attr_idx])
                    )
                    show_table.add_row(
                        [attr_nm]
                        + np.sum(atrribute_results[:, attr_idx], axis=0).tolist()
                        + [recall]
                    )
                show_table.add_row(
                    ["precision"]
                    + [
                        "{:0.3f}".format(
                            np.sum(atrribute_results[:, attr_idx, attr_idx])
                            / np.sum(atrribute_results[:, :, attr_idx])
                        )
                        for attr_idx in range(num_attr)
                    ]
                    + ["None"]
                    + ["None"]
                )
                print_log(
                    msg="====================TOTAL====================", logger=logger
                )
                print_log(msg=show_table, logger=logger)

            if logger is None:
                msg = "\n" + msg
            print_log(msg, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
