import torch
import torch.nn as nn
import torch.nn.functional as F


def BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()


class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return loss


class DiceLoss:
    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        )
        return loss.mean()


class IoULoss:
    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU


class BCEDiceLoss:
    def __init__(self, bce_weight=0.5):
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return loss


class MSELoss:
    def __call__(self, pred, target):
        return F.mse_loss(pred, target)
