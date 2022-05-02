import torch
import torch.nn as nn
from model.backbone import *
import numpy as np

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg, size=(280,800), pretrained=False, cls_dim=(37,10,4), use_aux=True):
        super().__init__()
        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.use_aux = use_aux

        self.cls_dim = cls_dim # num_gridding, num_row_anchor, num_of_lane

        self.total_dim = np.prod(cls_dim) # 37 * 10 * 4

        self.model = resnet(self.total_dim, 3)

        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(True),
            nn.Linear(2048, self.total_dim)
        )
        
        self.pool = nn.Conv2d(512, 8, 1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4

        # segmentation branch
        if self.use_aux:
            self.aux_header2 = nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_combine = nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                nn.Conv2d(128, cls_dim[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w
            )

    def forward(self, x):
        # n c h w = n 2048 sh sw
        # -> n 2048
        x2, x3, feature = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')
            x4 = self.aux_header4(x4)
            x4 = nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None
        
        feature = self.pool(feature).view(-1, 1800)
        group_cls = self.cls(feature).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg
        return group_cls