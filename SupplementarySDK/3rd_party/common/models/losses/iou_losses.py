import mmcv
import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def rank_iou_loss(pred, target, eps=1e-6):
    def calc_iou(pred, target, eps=1e-6):

        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        return ious

    ori_ious = calc_iou(pred, target, eps)
    dist_delta = torch.abs(pred - target)
    _, indices = torch.sort(dist_delta, dim=0, descending=True)
    rank_x1, rank_y1 = pred[:, 0][indices[:, 0]], pred[:, 1][indices[:, 1]]
    rank_x2, rank_y2 = pred[:, 2][indices[:, 2]], pred[:, 3][indices[:, 3]]
    rank_pred = torch.stack((rank_x1, rank_y1, rank_x2, rank_y2), dim=1)
    rank_ious = calc_iou(rank_pred, target, eps)

    ious = torch.max(ori_ious, rank_ious)
    loss = 1 - ious
    # loss = -ious.log()
    return loss


@LOSSES.register_module()
class RankIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(RankIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rank_iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss
