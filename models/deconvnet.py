import torch.nn as nn


class deconvnet(nn.Module):
    def __init__(self, num_classes, in_channels):
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
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.encoder = nn.ModuleList(
            [
                CBR(in_channels, 64, 3, 1, 1),
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
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True),
            ]
        )

        self.middle = nn.ModuleList(
            [
                CBR(512, 4096, 7, 1, 0),
                nn.Dropout2d(0.5),
                CBR(4096, 4096, 1, 1, 0),
                nn.Dropout2d(0.5),
                DCB(4096, 512, 7, 1, 0),
            ]
        )

        self.decoder = nn.ModuleList(
            [
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
            ]
        )

        self.score = nn.Conv2d(64, num_classes, 1, 1, 0, 1)

    def forward(self, x):
        h = x

        # encoder
        indices = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                h, pool_indices = layer(h)
                indices.append(pool_indices)
            else:
                h = layer(h)

        # middle
        for layer in self.middle:
            h = layer(h)

        # decoder
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                h = layer(h, indices.pop())
            else:
                h = layer(h)

        return self.score(h)

    def train_step(self, x):
        return self.forward(x)
