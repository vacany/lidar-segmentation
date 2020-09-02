import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, bias=bias)

        self.conv = DoubleConv(in_channels, out_channels, bias=bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=False):
        super(UNet, self).__init__()
        self.name = 'Unet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, bias=bias)
        self.down1 = Down(64, 128, bias=bias)
        self.down2 = Down(128, 256, bias=bias)
        self.down3 = Down(256, 256, bias=bias)
        self.up1 = Up(512, 256, bilinear, bias=bias)
        self.up2 = Up(384, 128, bilinear, bias=bias)
        self.up3 = Up(192, 64, bilinear, bias=bias)
        self.outc = OutConv(64, n_classes, bias=bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform_(lay.weight)
                #lay.bias.data.fill_(0.01)

class Lil_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=False):
        super(Lil_UNet, self).__init__()
        self.name = 'LilUnet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, bias=bias)
        self.down1 = Down(64, 128, bias=bias)
        self.down2 = Down(128, 256, bias=bias)

        self.up2 = Up(384, 128, bilinear, bias=bias)
        self.up3 = Up(192, 64, bilinear, bias=bias)
        self.outc = OutConv(64, n_classes, bias=bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform_(lay.weight)
                #lay.bias.data.fill_(0.01)
