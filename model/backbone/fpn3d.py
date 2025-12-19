import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 3D FPN  =====================
class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownSample3D(nn.Module):


    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=stride, padding=1)  # 时空下采样
        self.conv = Conv3DBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class FPN3D(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.bottom_up = nn.ModuleList([
            Conv3DBlock(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            DownSample3D(base_channels, base_channels * 2),
            DownSample3D(base_channels * 2, base_channels * 4),
            DownSample3D(base_channels * 4, base_channels * 8),
            DownSample3D(base_channels * 8, base_channels * 16)
        ])

        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(base_channels * 2, 256, kernel_size=1),
            nn.Conv3d(base_channels * 4, 256, kernel_size=1),
            nn.Conv3d(base_channels * 8, 256, kernel_size=1),
            nn.Conv3d(base_channels * 16, 256, kernel_size=1)
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.smooth_convs = nn.ModuleList([
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.Conv3d(256, 256, kernel_size=3, padding=1)
        ])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 全局时空平均池化

        self.fc_cls = nn.Sequential(
            nn.Linear(256 * 4, 512),  # 融合4个尺度特征（每个256维）
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, in_channels)
        )

    def forward(self, x):
        c1 = self.bottom_up[0](x)
        c2 = self.bottom_up[1](c1)
        c3 = self.bottom_up[2](c2)
        c4 = self.bottom_up[3](c3)
        c5 = self.bottom_up[4](c4)

        p5 = self.lateral_convs[3](c5)

        p4 = self.lateral_convs[2](c4)
        p5_up = self.upsample(p5)
        if p5_up.shape[-3:] != p4.shape[-3:]:
            p5_up = F.interpolate(p5_up, size=p4.shape[-3:], mode='trilinear', align_corners=False)
        p4 = p4 + p5_up

        p3 = self.lateral_convs[1](c3)
        p4_up = self.upsample(p4)
        if p4_up.shape[-3:] != p3.shape[-3:]:
            p4_up = F.interpolate(p4_up, size=p3.shape[-3:], mode='trilinear', align_corners=False)
        p3 = p3 + p4_up

        p2 = self.lateral_convs[0](c2)
        p3_up = self.upsample(p3)
        if p3_up.shape[-3:] != p2.shape[-3:]:
            p3_up = F.interpolate(p3_up, size=p2.shape[-3:], mode='trilinear', align_corners=False)
        p2 = p2 + p3_up

        p4 = self.smooth_convs[2](p4)
        p3 = self.smooth_convs[1](p3)
        p2 = self.smooth_convs[0](p2)


        f2 = self.avgpool(p2).flatten(1)  # [B, 256]
        f3 = self.avgpool(p3).flatten(1)  # [B, 256]
        f4 = self.avgpool(p4).flatten(1)  # [B, 256]
        f5 = self.avgpool(p5).flatten(1)  # [B, 256]

        x_feat = torch.cat([f2, f3, f4, f5], dim=1)
        x_cls = self.fc_cls(x_feat)  # [B, 3]
        return x_cls, x_feat


def get_fpn3d(args):
    return FPN3D(in_channels=3, base_channels=64)