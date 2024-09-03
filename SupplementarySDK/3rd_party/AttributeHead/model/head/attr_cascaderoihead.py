import numpy as np
import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import build_head, build_roi_extractor
from mmdet.models.roi_heads import CascadeRoIHead, StandardRoIHead

from ..builder import ATTR_HEADS
from .utils import bbox2result


@ATTR_HEADS.register_module()
class SingleAttrCascadeRoIHead(CascadeRoIHead):
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
        if self.with_mask:
            raise NotImplementedError("masks head is not suported")
        gt_attrs = kwargs["gt_attrs"]
        losses = dict()
        for i in range(self.num_stages):
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]
            sampling_results = []

            # assign gts and sample proposals
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_bboxes_ignore[j],
                        gt_labels[j],
                    )

                    attr_assign_result = bbox_assigner.assign(
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_bboxes_ignore[j],
                        gt_attrs[j],
                    )

                    assign_result.labels = torch.stack(
                        (assign_result.labels, attr_assign_result.labels), dim=1
                    )
                    gt_label_with_attr = torch.stack((gt_labels[j], gt_attrs[j]), dim=1)
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_label_with_attr,
                        feats=[lvl_feat[j][None] for lvl_feat in x],
                    )
                    sampling_results.append(sampling_result)

            bbox_results = self._bbox_forward_train(
                i, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg
            )
            for name, value in bbox_results["loss_bbox"].items():
                losses[f"s{i}.{name}"] = value * lw if "loss" in name else value

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results["bbox_targets"][0]
                with torch.no_grad():
                    cls_score = bbox_results["cls_score"]
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(cls_score)

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1),
                        roi_labels,
                    )
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results["rois"],
                        roi_labels,
                        bbox_results["bbox_pred"],
                        pos_is_gts,
                        img_metas,
                    )

        return losses

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        bbox_feats = bbox_roi_extractor(x[: bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred, attr_score = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            attr_score=attr_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
        )
        return bbox_results

    def _bbox_forward_train(
        self, stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg
    ):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        (
            labels,
            attr_targets,
            label_weights,
            bbox_targets,
            bbox_weights,
        ) = self.bbox_head[stage].get_targets(
            sampling_results=sampling_results,
            rcnn_train_cfg=rcnn_train_cfg,
        )
        loss_bbox = self.bbox_head[stage].loss(
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

        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            bbox_targets=(
                labels,
                attr_targets,
                label_weights,
                bbox_targets,
                bbox_weights,
            ),
        )
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, "Bbox head must be implemented."
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        # ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        ms_bbox_result = {}
        # ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            batch_size = len(proposal_list)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            det_attr = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros((0, self.bbox_head.fc_cls.out_features))
            return (
                [det_bbox] * batch_size,
                [det_label] * batch_size,
                [det_attr] * batch_size,
            )

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            cls_score = bbox_results["cls_score"]
            bbox_pred = bbox_results["bbox_pred"]
            attr_score = bbox_results["attr_score"]
            num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            attr_score = attr_score.split(num_proposals_per_img, 0)

            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s) for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j]
                        )
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_attrs  = []
        for i in range(num_imgs):
            det_bbox, det_label, det_attr = self.bbox_head[-1].get_bboxes(
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

        bbox_results = [
            bbox2result(
                det_bboxes[i],
                det_labels[i],
                det_attrs[i],
                self.bbox_head[-1].num_classes,
            )
            for i in range(num_imgs)
        ]
        ms_bbox_result["ensemble"] = bbox_results
        # with mask to do
        results = ms_bbox_result["ensemble"]
        return results

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        rois = bbox2roi(proposals)

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
