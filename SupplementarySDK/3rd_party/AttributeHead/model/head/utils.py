import numpy as np
import torch
from mmcv.ops.nms import batched_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps


def bbox2result(bboxes, labels, attrs, num_classes):
    if bboxes.shape[0] == 0:
        return [
            [np.zeros((0, 5), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)]
            for i in range(num_classes)
        ]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            attrs = attrs.detach().cpu().numpy()
        return [
            [bboxes[labels == i, :], attrs[labels == i, :]] for i in range(num_classes)
        ]


def multiclass_nms_with_attr(
    multi_bboxes,
    multi_scores,
    attr_scores,
    score_thr,
    nms_cfg,
    max_num=-1,
    score_factors=None,
    return_inds=False,
):
    num_classes = multi_scores.size(1) - 1

    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    attrs = attr_scores[:, None, :].expand(
        multi_scores.size(0), num_classes, attr_scores.size(1)
    )
    attrs = attrs.reshape(-1, attr_scores.size(1))

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes
        )
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        if attrs is not None:
            attrs = attrs[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, attrs, inds
        else:
            return dets, labels, attrs

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return (
            dets,
            labels[keep],
            inds[keep],
            attrs[keep],
        )
    else:
        return dets, labels[keep], attrs[keep]
