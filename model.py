import torch
from torchvision import models
import torch.nn as nn
import loss

class fcn_resnet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, config['class_num'], kernel_size=1)
    def forward(self, image):
        return self.model(image)['out']
    
    def train_step(self, image, masks):
        image = image.to(self.device)
        
        outputs = self.forward(image)
        return outputs