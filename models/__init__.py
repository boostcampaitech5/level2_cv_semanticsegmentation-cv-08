from .fcn import fcn8, fcn32, fcn_resnet50, fcn_resnet101
from .deeplab import deeplabv3_resnet50, deeplabv3_resnet101
from .deconvnet import deconvnet

__all__ = [
    "fcn_resnet50",
    "fcn_resnet101",
    "fcn32",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "fcn8",
    "deconvnet"
]
