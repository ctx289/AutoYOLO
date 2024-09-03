# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models.builder import build_loss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import ConvFCBBoxHead
from mmdet.models.utils import build_linear_layer
from torch.nn.modules.utils import _pair

from .utils import multiclass_nms_with_attr
from ..builder import ATTR_HEADS


@ATTR_HEADS.register_module()
class SingleAttrBBoxHead(ConvFCBBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively. @fined"""

    def __init__(
        self,
        num_attrs,
        use_back_bbox=False,
        loss_attr=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        attr_predictor_cfg=dict(type="Linear"),
        *args,
        **kwargs
    ):
        super(SingleAttrBBoxHead, self).__init__(*args, **kwargs)
        self.use_back_bbox = use_back_bbox

        self.loss_attr = loss_attr
        self.num_attrs = num_attrs

        self.attr_predictor_cfg = attr_predictor_cfg
        self.loss_attr = build_loss(loss_attr)

        if self.custom_attr_channels:
            attr_channels = self.loss_attr.get_cls_channels(self.num_attrs)
        else:
            if self.use_back_bbox:
                attr_channels = num_attrs + 1
            else:
                attr_channels = num_attrs

        self.fc_attr = build_linear_layer(
            self.attr_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=attr_channels,
        )

        if "init_cfg" not in kwargs or kwargs["init_cfg"] is None:
            self.init_cfg += [
                dict(type="Normal", std=0.001, override=dict(name="fc_attr"))
            ]

    @property
    def custom_attr_channels(self):
        return getattr(self.loss_attr, "custom_cls_channels", False)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # shared neck for cls and attr
        cls_score = self.fc_cls(x_cls)
        bbox_pred = self.fc_reg(x_reg)
        attr_score = self.fc_attr(x_cls)

        return cls_score, bbox_pred, attr_score

    def _get_target_single(
        self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, pos_gt_attrs, cfg
    ):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        attrs = pos_bboxes.new_full((num_samples,), self.num_attrs, dtype=torch.long)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            attrs[:num_pos] = pos_gt_attrs
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, attrs, label_weights, bbox_targets, bbox_weights

    def get_targets(
        self,
        sampling_results,
        rcnn_train_cfg,
        concat=True,
    ):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]

        pos_gt_labels_list = [res.pos_gt_labels[:, 0] for res in sampling_results]
        pos_gt_attrs_list = [res.pos_gt_labels[:, 1] for res in sampling_results]

        labels, attrs, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_attrs_list,
            cfg=rcnn_train_cfg,
        )

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            attr_targets = torch.cat(attrs, 0)

        return (
            labels,
            attr_targets,
            label_weights,
            bbox_targets,
            bbox_weights,
        )

    @force_fp32(apply_to=("cls_score", "bbox_pred", "attr_score"))
    def loss(
        self,
        cls_score,
        attr_score,
        bbox_pred,
        rois,
        labels,
        attr_targets,
        label_weights,
        bbox_targets,
        bbox_weights,
        reduction_override=None,
    ):
        losses = super(SingleAttrBBoxHead, self).loss(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            rois=rois,
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            reduction_override=reduction_override,
        )
        if attr_score.numel() > 0:
            if self.use_back_bbox:
                valid_inds = (attr_targets >= 0)
            else:
                valid_inds = (attr_targets >= 0) & (attr_targets != self.num_attrs)
            # # avg_factor = torch.sum(label_weights[valid_inds] > 0).float().item()
            # valid_inds = (attr_targets >= 0)
            avg_factor = max(
                torch.sum(label_weights[valid_inds] > 0).float().item(), 1.0
            )
            loss_attr_ = self.loss_attr(
                attr_score[valid_inds],
                attr_targets[valid_inds],
                label_weights[valid_inds],
                avg_factor=avg_factor,
                reduction_override=reduction_override,
            )
            if isinstance(loss_attr_, dict):
                losses.update(loss_attr_)
            else:
                losses["loss_attr"] = loss_attr_
            if self.custom_activation:
                acc_attr_ = self.loss_attr.get_accuracy(attr_score, attr_targets)
                losses.update(acc_attr_)
            else:
                losses["acc_attr"] = accuracy(
                    attr_score[valid_inds], attr_targets[valid_inds]
                )
        return losses

    @force_fp32(apply_to=("cls_score", "bbox_pred", "attr_score"))
    def get_bboxes(
        self,
        rois,
        cls_score,
        attr_score,
        bbox_pred,
        img_shape,
        scale_factor,
        rescale=False,
        cfg=None,
    ):
        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        if self.custom_attr_channels:
            attrs = self.loss_attr.get_activation(attr_score)
        else:
            attrs = F.softmax(attr_score, dim=-1)

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape
            )
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1
            )

        if cfg is None:
            return bboxes, scores, attrs
        else:
            det_bboxes, det_labels, det_attrs = multiclass_nms_with_attr(
                bboxes,
                scores,
                attrs,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
            )
            return det_bboxes, det_labels, det_attrs


@ATTR_HEADS.register_module()
class Shared2FCBBoxHead(SingleAttrBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs
        )


@ATTR_HEADS.register_module()
class Shared4Conv1FCBBoxHead(SingleAttrBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs
        )
