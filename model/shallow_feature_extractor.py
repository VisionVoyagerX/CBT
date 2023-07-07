import torch.nn as nn
from .image_reconstructor import *


class Shallow_Feature_Extractor(nn.Module):
    """Shallow Feature Extraction module

    Args:
        num_in_ch (int): Number of input channels.
        embed_dim (int): Dimension of embedding.
        kernel_size (int): Kernel size.
        stride (int): Stride size.
        padding (int): Padding size.
        upsample (bool): Enable upsampling.
        pan_low_size_ratio (int): Ratio of size difference between a pan-sharpened image and a low-resolution image.
    """
    def __init__(self,num_in_ch, embed_dim, kernel_size=3, stride=1, padding=1, upsample=False, pan_low_size_ratio=None):
        super().__init__()
        self.upsample = upsample
        if self.upsample == True:
            self.upsampler = Image_Reconstruction(num_in_ch, embed_dim, embed_dim, pan_low_size_ratio)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, kernel_size, stride, padding)

    def forward(self, x):
        if self.upsample == True:
            x = self.upsampler(x)
        else:
            x = self.conv_first(x)
        return x