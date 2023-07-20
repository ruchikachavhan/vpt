#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import logging
logger = logging.get_logger("visual_prompt")

class MSELoss(nn.Module):
    def __init__(self, cfg=None):
        super(MSELoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def is_single(self):
        return True
    
    def is_local(self):
        return False
    
    def loss(self, pred, target):
        # return self.cos(pred, target)
        
        return F.l1_loss(pred, target)
    
    def forward(self, pred, target, per_cls_weights=None):
        loss = self.loss(pred, target).mean()
        return loss

class KLD(nn.Module):
    def __init__(self, cfg=None):
        super(KLD, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        # According to https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
        #  We do log_softmax of input and softmax of target
    
    def is_single(self):
        return True
    
    def is_local(self):
        return False
    
    def loss(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        target = F.softmax(target, dim=1)
        return self.kl_div(pred, target)
    
    def forward(self, pred, target, per_cls_weights=None):
        loss = self.loss(pred, target)
        return loss

class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        return torch.sum(loss) / targets.shape[0]


LOSS = {
    "softmax": SoftmaxLoss,
    "mse": MSELoss,
    "kl": KLD,
    "cross_entropy": SoftmaxLoss,
}


def build_loss(cfg):
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
