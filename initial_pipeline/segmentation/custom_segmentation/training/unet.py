import torch.nn as nn
import torch.nn.functional as F
import torch



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.u2 = DoubleConv(256 + 128, 128)
        self.u1 = DoubleConv(128 + 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(F.max_pool2d(c1, 2))
        c3 = self.d3(F.max_pool2d(c2, 2))

        u2 = F.interpolate(c3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, c2], dim=1))

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, c1], dim=1))

        return self.out(u1)