from .hrnet import hrnet
from .deconvnet import deconvnet
from .deeplab import deeplabv3_resnet50, deeplabv3_resnet101
from .fcn import fcn8, fcn32, fcn_resnet50, fcn_resnet101

__all__ = [
    "fcn_resnet50",
    "fcn_resnet101",
    "fcn32",
    "fcn8",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deconvnet",
    "hrnet",
]
