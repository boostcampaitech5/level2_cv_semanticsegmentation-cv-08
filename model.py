import torch
from torchvision import models
import torch.nn as nn
import loss
from einops import rearrange
# from torchvision.ops import StochasticDepth
from typing import List, Iterable
import torch.nn.functional as F
import segmentation_models_pytorch as smp

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

# 아래 모든 module은 segFormer를 위한 module 입니다.
# layerNorm은 batch, h, w, c의 shape에서 동작하기 때문에 채널의 순서를 변경해준다.
# 아래 class에서는 einops라는 module을 이용해서 쉽게 변경한다.
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out
    
class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )             

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    # StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )
        
class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        
        # super().__init__()
        # self.overlap_patch_merge = OverlapPatchMerging(
        #     in_channels, out_channels, patch_size, overlap_size,
        # )
        # self.blocks = nn.Sequential(
        #     *[
        #         SegFormerEncoderBlock(
        #             out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
        #         )
        #         for i in range(depth)
        #     ]
        # )
        # self.norm = LayerNorm2d(out_channels)
        
        super().__init__(
            OverlapPatchMerging(
                in_channels, out_channels, patch_size, overlap_size,
            ),
            nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]),
            LayerNorm2d(out_channels)
        )
        
def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk
        
class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class SegFormer(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int=3,
        widths: List[int]=[64, 128, 256, 512],
        depths: List[int]=[3, 4, 6, 3],
        all_num_heads: List[int]=[1, 2, 4, 8],
        patch_sizes: List[int]=[7, 3, 3, 3],
        overlap_sizes: List[int]=[4, 2, 2, 2],
        reduction_ratios: List[int]=[8, 4, 2, 1],
        mlp_expansions: List[int]=[4, 4, 4, 4],
        decoder_channels: int=256,
        scale_factors: List[int]=[8, 4, 2, 1],
        num_classes: int=29,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features= self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation

    def train_step(self, x):
        h = self.forward(x)
        out = F.interpolate(h, size=(self.config['feed_size'], self.config['feed_size']), mode='bilinear', align_corners=True)
        return out
        
class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights ='imagenet',
            classes=config['class_num']
        )
    
    def forward(self, x):
        return self.model.forward(x)

    def train_step(self, x):
        return self.forward(x)
    
if __name__=="__main__":
    import argparse
    from parse_config import ConfigParser
    
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='./config.json')
    config = ConfigParser.from_args(args)
    
    net = Unet(config)
    
    # net = net(torch.randn((1, 3, 512, 512)))
    for p in net.parameters():
        print(p)