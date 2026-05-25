import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet (nn.module):
    def __init__(Self, in_channels, num_classes):
        super().__init__()
        self.down_convulation_1 = DownSample(in_channels, 64)
        self.down_convulation_2 = DownSample(in_channels, 128)


