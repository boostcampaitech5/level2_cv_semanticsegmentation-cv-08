from .deconvnet import deconvnet
from .deeplab import deeplabv3_resnet50, deeplabv3_resnet101
from .fcn import fcn8, fcn32, fcn_resnet50, fcn_resnet101
from .hrnet import hrnet

__all__ = [
    "fcn_resnet50",
    "fcn_resnet101",
    "fcn32",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "fcn8",
    "deconvnet",
    "hrnet",
]
