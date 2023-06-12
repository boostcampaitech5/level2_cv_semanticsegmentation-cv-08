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
    
    def train_step(self, image):
        image = image.to(self.device)
        
        outputs = self.forward(image)
        return outputs
    
class fcn32(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.ReLU(inplace=True)
            )
        
        self.nets = nn.ModuleList([
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
            
            nn.Conv2d(4096, config['class_num'], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(config['class_num'], config['class_num'], kernel_size=64, stride=32, padding=16)
        ])
        
    def forward(self, x):
        h = x
        for net in self.nets:
            h = net(h)
        return h
    
    def train_step(self, images):
        h = self.forward(images)
        return h

class FCN8s(nn.Module):
    def __init__(self, config):
        super(FCN8s, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, 
                                            padding=padding),
                                  nn.ReLU(inplace=True)
                                 )        
        
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
        
        # conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)  
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)          
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256,
                                        config['class_num'], 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)             
        
        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)          
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Score pool4
        self.score_pool4_fr = nn.Conv2d(512,
                                        config['class_num'], 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)        
        
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
    
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()


        # Score
        self.score_fr = nn.Conv2d(4096, config['class_num'], kernel_size = 1)
        
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(config['class_num'],
                                           config['class_num'],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        # UpScore2_pool4 using deconv
        self.upscore2_pool4 = nn.ConvTranspose2d(config['class_num'], 
                                                 config['class_num'], 
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)
        
        # UpScore8 using deconv
        self.upscore8 = nn.ConvTranspose2d(config['class_num'], 
                                           config['class_num'],
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)


    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)        
        pool3 = h = self.pool3(h)

        # Score
        score_pool3c = self.score_pool3_fr(pool3)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        pool4 = h = self.pool4(h)

        # Score
        score_pool4c = self.score_pool4_fr(pool4)       
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h = self.pool5(h)
        
        h = self.fc6(h)
        h = self.relu6(h) # update
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.relu7(h) # update
        h = self.drop7(h)
        
        h = self.score_fr(h)
        
        # Up Score I
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)
        
        # Sum II
        h = upscore2_pool4c + score_pool3c
        
        # Up Score III
        upscore8 = self.upscore8(h)
        
        return upscore8
    
    def train_step(self, images):
        h = self.forward(images)
        return h
    
class DeconvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        
        def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        
        self.encoder = nn.ModuleList([
            CBR(3, 64, 3, 1, 1),
            CBR(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True),
            
            CBR(64, 128, 3, 1, 1),
            CBR(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True),
            
            CBR(128, 256, 3, 1, 1),
            CBR(256, 256, 3, 1, 1),
            CBR(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True),
            
            CBR(256, 512, 3, 1, 1),
            CBR(512, 512, 3, 1, 1),
            CBR(512, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True),
            
            CBR(512, 512, 3, 1, 1),
            CBR(512, 512, 3, 1, 1),
            CBR(512, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
        ])
        
        self.middle = nn.ModuleList([
            CBR(512, 4096, 7, 1, 0),
            nn.Dropout2d(0.5),
            CBR(4096, 4096, 1, 1, 0),
            nn.Dropout2d(0.5),
            DCB(4096, 512, 7, 1, 0)])
        
        self.decoder = nn.ModuleList([
            nn.MaxUnpool2d(2, stride=2),
            DCB(512, 512, 3, 1, 1),
            DCB(512, 512, 3, 1, 1),
            DCB(512, 512, 3, 1, 1),
            
            nn.MaxUnpool2d(2, stride=2),
            DCB(512, 512, 3, 1, 1),
            DCB(512, 512, 3, 1, 1),
            DCB(512, 256, 3, 1, 1),
            
            nn.MaxUnpool2d(2, stride=2),
            DCB(256, 256, 3, 1, 1),
            DCB(256, 256, 3, 1, 1),
            DCB(256, 128, 3, 1, 1),
            
            nn.MaxUnpool2d(2, stride=2),
            DCB(128, 128, 3, 1, 1),
            DCB(128, 64, 3, 1, 1),
            
            nn.MaxUnpool2d(2, stride=2),
            DCB(64, 64, 3, 1, 1),
            DCB(64, 64, 3, 1, 1),
        ])
        
        self.score = nn.Conv2d(64, config['class_num'], 1, 1, 0, 1)
        
    def forward(self, x):
        h = x
        
        #encoder
        indices = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                h, pool_indices = layer(h)
                indices.append(pool_indices)
            else:
                h = layer(h)
        
        #middle
        for layer in self.middle:
            h = layer(h)
            
        #decoder
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                h = layer(h, indices.pop())
            else:
                h = layer(h)
        
        return self.score(h)
    
    def train_step(self, x):
        return self.forward(x)
    

if __name__=="__main__":
    import argparse
    from parse_config import ConfigParser
    
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='./config.json')
    config = ConfigParser.from_args(args)
    
    net = DeconvNet(config)
    for l in net.children():
        print(l)