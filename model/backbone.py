import torch
import torch.nn as nn
from typing import cast

cfg : dict = {
  'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def make_layers(cfg : dict = cfg["D"], in_channels : int = 3):
    layers = nn.ModuleList()

    for v in cfg:
        # max pooling
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # batch_norm
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # activation
            layers += [conv2d, nn.ReLU(inplace=True)]

            # 다음 conv input channel 
            in_channels = v
    return nn.Sequential(*layers)


model_list = [
    "vgg",
    "vgg16",
    "vgg16_bn",
]

class VGG(nn.Module):
    def __init__(self, 
                total_dim : int, in_channels : int = 3, 
                is_train : bool = False, init_weight : bool = False) -> None:
        super().__init__()
        self.total_dim = total_dim
        self.in_channels = in_channels
        self.is_train = is_train
        self.layers = make_layers()
        # self.weights = "https://download.pytorch.org/models/vgg16-6c64b313.pth"

        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifer = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096,4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, total_dim)
        )

        if init_weight:
            self._initalize_weights()
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x

    def _initalize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ---------------------------- #

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x += self.downsample(x)
        return x

class resnet(nn.Module):
    def __init__(self, out_channels : int, in_channels : int = 3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.initialize_weight(self.make_layer(64, 64, stride=1))
        self.layer2 = self.initialize_weight(self.make_layer(64, 128, stride=2))
        self.layer3 = self.initialize_weight(self.make_layer(128, 256, stride=2))
        self.layer4 = self.initialize_weight(self.make_layer(256, 512, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_channels)

    def make_layer(self, in_channels : int, out_channels : int, stride : int):
        strides = stride + 1
        layers = []
        for stride in range(strides):
            layers.append(BasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x = self.avgpool(x4)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x2,x3,x4


    def initalize_weights(*models) -> None:
        for m in models:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)