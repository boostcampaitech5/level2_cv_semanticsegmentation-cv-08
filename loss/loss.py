import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .msssimLoss import MSSSIM

def BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()


class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return loss


class DiceLoss:
    def __init__(self, smooth=1.0, sigmoid=True):
        self.smooth = smooth
        self.sigmoid = sigmoid

    def __call__(self, pred, target):
        if self.sigmoid:
            pred = F.sigmoid(pred)
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        )
        return loss.mean()


class IoULoss:
    def __init__(self, smooth=1.0, sigmoid=True):
        self.smooth = smooth
        self.sigmoid = sigmoid

    def __call__(self, inputs, targets):
        if self.sigmoid:
            inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        IoU = torch.mean(IoU)
        return 1 - IoU


class BCEDiceLoss:
    def __init__(self, bce_weight=0.5):
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(sigmoid=False)

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return loss
    
class UNet3pHybridLoss:
    def __init__(self):
        self.focal_loss = FocalLoss()
        self.Iou_loss = IoULoss()
        self.msssim_loss = MSSSIM()
    
    def __call__(self, pred, target):
        focal = self.focal_loss(pred, target)
        iou = self.Iou_loss(pred, target)
        pred = F.sigmoid(pred)
        msssim_list = []
        for i in range(29):
            msssim_list.append(self.msssim_loss(pred[:, i:i+1, :, :], target[:, i:i+1, :, :]))
        msssim = 1 - torch.mean(torch.stack(msssim_list))
        print("focal : ", focal.item(), ", iou : ", iou.item(), ", msssim : ", msssim.item())
        # print("focal : ", focal.item(), ", iou : ", iou.item())
        loss = focal + iou # + msssim
        return loss


class BCEDiceIoULoss:
    def __init__(self, bce_weight=0.8, dice_weight=1.0, iou_weight=1.0):
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.dice_loss = DiceLoss(sigmoid=False)
        self.iou_loss = IoULoss(sigmoid=False)

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        iou = self.iou_loss(pred, target)
        loss = bce * self.bce_weight + dice * self.dice_weight + iou * self.iou_weight
        return loss


class MSELoss:
    def __call__(self, pred, target):
        pred = F.sigmoid(pred)
        return F.mse_loss(pred, target)
