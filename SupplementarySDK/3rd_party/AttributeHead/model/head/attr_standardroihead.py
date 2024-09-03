import numpy as np
import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead

from .utils import bbox2result
from ..builder import ATTR_HEADS


@ATTR_HEADS.register_module()
class SingleAttrStandardRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head. @fined"""

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        **kwargs,
    ):
        # assign gts and sample proposals
        assert "gt_attrs" in kwargs, "groudtruth attributes is not assigned"
        gt_attrs = kwargs["gt_attrs"]
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_bboxes_ignore[i],
                    gt_labels[i],
                )
                attr_assign_result = self.bbox_assigner.assign(
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_bboxes_ignore[i],
                    gt_attrs[i],
                )

                # combine the gt_labels and the attr_labels
                assign_result.labels = torch.stack(
                    (assign_result.labels, attr_assign_result.labels), dim=1
                )
                gt_label_with_attr = torch.stack((gt_labels[i], gt_attrs[i]), dim=1)

                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_label_with_attr,
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results)
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_masks, img_metas
            )
            losses.update(mask_results["loss_mask"])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, attr_score = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            attr_score=attr_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
        )
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        (
            labels,
            attr_targets,
            label_weights,
            bbox_targets,
            bbox_weights,
        ) = self.bbox_head.get_targets(
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg,
        )

        loss_bbox = self.bbox_head.loss(
            cls_score=bbox_results["cls_score"],
            attr_score=bbox_results["attr_score"],
            bbox_pred=bbox_results["bbox_pred"],
            rois=rois,
            labels=labels,
            attr_targets=attr_targets,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
        )
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, "Bbox head must be implemented."

        det_bboxes, det_labels, det_attrs = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        bbox_results = [
            bbox2result(
                det_bboxes[i],
                det_labels[i],
                det_attrs[i],
                self.bbox_head.num_classes,
            )
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        elif self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale
            )
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            det_attr = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros((0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return (
                [det_bbox] * batch_size,
                [det_label] * batch_size,
                [det_attr] * batch_size,
            )

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results["cls_score"]
        bbox_pred = bbox_results["bbox_pred"]
        attr_score = bbox_results["attr_score"]

        num_proposals_per_img = tuple(len(p) for p in proposals)

        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        attr_score = attr_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_attrs = []

        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                det_attrs = rois[i].new_zeros((0,), dtype=torch.long)

                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features)
                    )
                    det_attrs = rois[i].new_zeros(
                        (0, self.bbox_head.fc_attr.out_features)
                    )
            else:
                det_bbox, det_label, det_attr = self.bbox_head.get_bboxes(
                    rois=rois[i],
                    cls_score=cls_score[i],
                    attr_score=attr_score[i],
                    bbox_pred=bbox_pred[i],
                    img_shape=img_shapes[i],
                    scale_factor=scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,
                )

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_attrs.append(det_attr)

        return det_bboxes, det_labels, det_attrs
