from .deconvnet import deconvnet
from .deeplab import deeplabv3_resnet50, deeplabv3_resnet101
from .fcn import fcn8, fcn32, fcn_resnet50, fcn_resnet101, fcn8_7d
from .hrnet import hrnet
from .unet import UNet_3Plus_DeepSup_CGM, UNet_3Plus, UNet_3Plus_DeepSup
__all__ = [
    "fcn_resnet50",
    "fcn_resnet101",
    "fcn32",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "fcn8",
    "deconvnet",
    "hrnet",
    "UNet_3Plus",
    "UNet_3Plus_DeepSup",
    "UNet_3Plus_DeepSup_CGM",
    "fcn8_7d"
]
