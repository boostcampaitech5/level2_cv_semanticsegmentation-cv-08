import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from utils import CLASS2IND
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

# class DiceLoss:
#     def __init__(self, smooth=1.0, sigmoid=True, class_weight=False):
#         self.smooth = smooth
#         self.sigmoid = sigmoid
#         self.class_weight = torch.ones([29]).cuda()
#         if class_weight:
#             for c, w in class_weight.items():
#                 self.class_weight[CLASS2IND[c]] = w

#     def __call__(self, pred, target):
#         if self.sigmoid:
#             pred = F.sigmoid(pred)    
#         pred = pred.contiguous()
#         target = target.contiguous()
#         intersection = (pred * target).sum(dim=2).sum(dim=2)
#         loss = 1 - (
#             (2.0 * intersection + self.smooth)
#             / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
#         )
#         print
#         loss.mul_(self.class_weight)
#         return loss.mean()


# class BCEDiceLoss:
#     def __init__(self, bce_weight=0.5, class_weight=False):
#         self.bce_weight = bce_weight
#         self.class_weight = torch.ones([29]).cuda()
#         if class_weight:
#             for c, w in class_weight.items():
#                 self.class_weight[CLASS2IND[c]] = w
#         self.dice_loss = DiceLoss(sigmoid=False, class_weight=class_weight)

#     def __call__(self, pred, target):
#         pred_flatten = pred.view(pred.size(0), -1, pred.size(1))
#         target_flatten = target.view(target.size(0), -1, target.size(1))
#         bce = F.binary_cross_entropy_with_logits(pred_flatten, target_flatten,weight=self.class_weight, pos_weight=self.class_weight)
#         pred = F.sigmoid(pred)
#         dice = self.dice_loss(pred, target)
#         loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
#         return loss

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

class BCEDiceFocalLoss:
    def __init__(self, bce_weight=1., dice_weight=1., focal_weight=1.):
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(sigmoid=False)
        self.focal_loss = FocalLoss()

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        focal = self.focal_loss(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = (bce * self.bce_weight + dice * self.dice_weight + focal * self.focal_weight) / (self.bce_weight + self.dice_weight + self.focal_weight)
        return loss    


class MSELoss:
    def __call__(self, pred, target):
        pred = F.sigmoid(pred)
        return F.mse_loss(pred, target)
