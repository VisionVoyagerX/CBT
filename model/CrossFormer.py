import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .shallow_feature_extractor import *
from .deep_feature_extractor import *
from .image_reconstructor import *


import matplotlib.pyplot as plt


class CrossFormer(nn.Module):
    r""" Hybrid Attention Transformer
        A PyTorch implementation of : `Activating More Pixels in Image Super-Resolution Transformer`.
        Some codes are based on SwinIR.
    Args:
        pan_img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        pan_low_size_ratio (int): Ratio of size difference between a pan-sharpened image and a low-resolution image. Default: 4
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 pan_img_size = 64,
                 pan_low_size_ratio = 4,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(CrossFormer, self).__init__()


        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.upscale = upscale
        self.upsampler = upsampler

        self.img_range = img_range

        self.mslr_mean = kwargs.get('mslr_mean')
        self.mslr_std =  kwargs.get('mslr_std')
        self.pan_mean =  kwargs.get('pan_mean')
        self.pan_std =  kwargs.get('pan_std')

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.pan_shallow_feature_extractor = Shallow_Feature_Extractor(1, embed_dim)
        self.mslr_shallow_feature_extractor = Shallow_Feature_Extractor(num_in_ch, embed_dim, upsample=True, pan_low_size_ratio = pan_low_size_ratio)
        
        # ------------------------- 2, deep feature extraction ------------------------- #

    
        self.pan_mslr_deep_feature_extractor = Deep_Feature_Extractor(pan_img_size,
                 patch_size,
                 in_chans,
                 embed_dim,
                 depths,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio,
                 qkv_bias,
                 qk_scale,
                 drop_rate,
                 attn_drop_rate,
                 drop_path_rate,
                 norm_layer,
                 ape,
                 patch_norm,
                 upscale,
                 img_range,
                 upsampler,
                 resi_connection,
                 )
        
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        self.image_recunstruction = Image_Reconstruction(embed_dim, num_feat, num_out_ch, upscale)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pan, mslr):
        #channel-wise normalization
        pan = (pan - self.pan_mean) / self.pan_std
        mslr = (mslr - self.mslr_mean) / self.mslr_std
        
        # shallow_feature_extractor
        pan = self.pan_shallow_feature_extractor(pan)
        mslr = self.mslr_shallow_feature_extractor(mslr)
        # deep_feature_extractor
        pan, mslr = self.pan_mslr_deep_feature_extractor(pan, mslr)
        #add
        mssr = pan + mslr  # FIXME should i concat (preserve all details) or add here (better gradient flow)?
        #image_reconstruction
        mssr = self.image_recunstruction(mssr)
        
        #channel-wise denormalization
        pan = pan * self.pan_std + self.pan_mean
        mssr = mssr * self.mslr_std + self.mslr_mean
    
        return mssr

if __name__ == "__main__":
    upscale = 3
    window_size = 8
    height = 32 #(95 // upscale // window_size + 1) * window_size
    width = 32 # (95 // upscale // window_size + 1) * window_size
    model = CrossFormer(upscale=1, pan_img_size=(height, width), pan_low_size_ratio= 3,
                window_size=window_size, img_range=1., depths=[6, 6, 6],
                embed_dim=30, num_heads=[6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')

    mslr = torch.rand((1, 3, height, width), dtype=torch.float32)
    pan = torch.rand((1, 1, height * 3, width * 3), dtype=torch.float32)

    print(height, ', ', width)
    print(model(pan, mslr).shape)
