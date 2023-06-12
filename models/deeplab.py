import torch
import torch.nn as nn
from torchvision import models

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
