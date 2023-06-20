import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn

from .swin import SwinTransformer


class SwinEncoder(torch.nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels = [3, 192, 384, 768, 1536]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 4

        self._in_channels: int = 3
        kwargs.pop("depth")

        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor):
        out_idn = nn.Identity()(x)
        outs = self.model(out_idn)
        return [out_idn, *outs]

        # x = self.model(x)
        # return list(x)

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict["model"], strict=False, **kwargs)


# swin_base

# def register_encoder():
#     smp.encoders.encoders["swin_encoder"] = {
#     "encoder": SwinEncoder,
#     "pretrained_settings": {
#         "imagenet": {
#             "mean": [0.485, 0.456, 0.406],
#             "std": [0.229, 0.224, 0.225],
#             "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
#             "input_space": "RGB",
#             "input_range": [0, 1],
#         },
#     },
#     "params": {
#         "pretrain_img_size": 384,
#         "embed_dim": 128,
#         "depths": [2, 2, 18, 2],
#         'num_heads': [4, 8, 16, 32],
#         "window_size": 12,
#         "drop_path_rate": 0.3,
#     }
# }


# swin_L
def register_encoder():
    smp.encoders.encoders["swin_encoder"] = {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "pretrain_img_size": 384,
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 12,
            "drop_path_rate": 0.3,
        },
    }
