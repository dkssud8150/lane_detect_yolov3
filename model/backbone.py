import torch
import torch.nn as nn

model_list = [
    "vgg",
    "vgg16",
    "vgg16_bn",
]

class VGG(nn.Module):
    def __init__(self, batch, n_classes, in_channels=3, in_width=800, in_height=280, is_train=False) -> None:
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        self.weights = "https://download.pytorch.org/models/vgg16-6c64b313.pth"

        
