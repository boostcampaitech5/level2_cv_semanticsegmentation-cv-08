import torch
import torch.nn as nn
from torchvision import models


class fcn_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, image):
        return self.model(image)["out"]

    def train_step(self, image):
        image = image.to(self.device)

        outputs = self.forward(image)
        return outputs

class fcn_resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.segmentation.fcn_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, image):
        return self.model(image)["out"]

    def train_step(self, image):
        image = image.to(self.device)

        outputs = self.forward(image)
        return outputs
    
class deeplabv3_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, image):
        return self.model(image)["out"]

    def train_step(self, image):
        image = image.to(self.device)

        outputs = self.forward(image)
        return outputs        

class deeplabv3_resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, image):
        return self.model(image)["out"]

    def train_step(self, image):
        image = image.to(self.device)

        outputs = self.forward(image)
        return outputs
    
class fcn32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
            )

        self.nets = nn.ModuleList(
            [
                CBR(3, 64, 3, 1, 1),
                CBR(64, 64, 3, 1, 1),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                CBR(64, 128, 3, 1, 1),
                CBR(128, 128, 3, 1, 1),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                CBR(128, 256, 3, 1, 1),
                CBR(256, 256, 3, 1, 1),
                CBR(256, 256, 3, 1, 1),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                CBR(256, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                CBR(512, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                CBR(512, 4096, 1, 1, 0),
                nn.Dropout2d(),
                CBR(4096, 4096, 1, 1, 0),
                nn.Dropout2d(),
                nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0),
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16),
            ]
        )

    def forward(self, x):
        h = x
        for net in self.nets:
            h = net(h)
        return h

    def train_step(self, images):
        h = self.forward(images)
        return h


# class fcn8(nn.Module):
#     def __init__(self, num_classes):

if __name__ == "__main__":
    print(__name__)
    import argparse

    from ..utils import num_classesParser

    args = argparse.ArgumentParser()
    args.add_argument("-c", "--num_classes", default="./num_classes.json")
    num_classes = num_classesParser.from_args(args)

    net = fcn_resnet50(num_classes)
    print(net)
