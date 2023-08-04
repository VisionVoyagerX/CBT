import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torchinfo import summary

from .shallow_feature_extractor import *
from .deep_feature_extractor import *
from .image_reconstructor import *


import matplotlib.pyplot as plt


class MBA(nn.Module):

    def __init__(self,
                 pan_img_size=64,
                 pan_low_size_ratio=4,
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
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        """ Multi Band Attention

        Args:
            pan_img_size (int, optional): Panchromatic band image size. Defaults to 64.
            pan_low_size_ratio (int, optional): Panchromatic to Low Resolution band ratio. Defaults to 4.
            patch_size (int, optional): Patch size. Defaults to 1.
            in_chans (int, optional): Input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 96.
            depths (tuple, optional): Number of HAB and CBAB in each MBAG. Defaults to (6, 6, 6, 6).
            num_heads (tuple, optional): Number of Attention heads in HAB and CBAB in each MBAG. Defaults to (6, 6, 6, 6).
            window_size (int, optional): Window size. Defaults to 7.
            compress_ratio (int, optional): Compress ratio. Defaults to 3.
            squeeze_factor (int, optional): Squeeze factor. Defaults to 30.
            conv_scale (float, optional): Conv scale. Defaults to 0.01.
            overlap_ratio (float, optional): Overlap ratio. Defaults to 0.5.
            mlp_ratio (_type_, optional): MLP ratio. Defaults to 4..
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            drop_rate (_type_, optional): Dropout rate. Defaults to 0..
            attn_drop_rate (_type_, optional): Dropout rate. Default: 0
            drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1
            norm_layer (_type_, optional): Normalization layer. Default: nn.LayerNorm.
            ape (bool, optional): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool, optional): If True, add normalization after patch embedding. Default: True
            upscale (int, optional): 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
            upsampler (str, optional): The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
            resi_connection (str, optional): The convolutional block before residual connection. '1conv'/'3conv'
        """
        super(MBA, self).__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.upscale = upscale
        self.upsampler = upsampler

        self.mslr_mean = kwargs.get('mslr_mean')
        self.mslr_std = kwargs.get('mslr_std')
        self.pan_mean = kwargs.get('pan_mean')
        self.pan_std = kwargs.get('pan_std')

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.pan_shallow_feature_extractor = Shallow_Feature_Extractor(
            1, embed_dim)
        self.mslr_shallow_feature_extractor = Shallow_Feature_Extractor(
            num_in_ch, embed_dim, upsample=True, pan_low_size_ratio=pan_low_size_ratio)

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
                                                                      upsampler,
                                                                      resi_connection,
                                                                      )

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        # FIXME change *2
        self.image_reconstruction = Image_Reconstruction(
            embed_dim * 2, num_feat, num_out_ch, upscale)

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
        # channel-wise normalization
        pan = (pan - self.pan_mean) / self.pan_std
        mslr = (mslr - self.mslr_mean) / self.mslr_std

        # shallow_feature_extractor
        pan = self.pan_shallow_feature_extractor(pan)
        mslr = self.mslr_shallow_feature_extractor(mslr)
        # deep_feature_extractor
        pan, mslr = self.pan_mslr_deep_feature_extractor(pan, mslr)
        # add
        # mssr = pan + mslr
        mssr = torch.concat((pan, mslr), dim=1)
        print('shape: ', mssr.shape)
        # image_reconstruction
        mssr = self.image_reconstruction(mssr)

        # channel-wise denormalization
        pan = pan * self.pan_std + self.pan_mean
        mssr = mssr * self.mslr_std + self.mslr_mean

        return mssr


if __name__ == "__main__":
    upscale = 4
    window_size = 8
    height = 16  # (95 // upscale // window_size + 1) * window_size
    width = 16  # (95 // upscale // window_size + 1) * window_size
    # precomputed
    pan_mean = torch.tensor([250.0172]).view(1, 1, 1, 1)
    pan_std = torch.tensor([80.2501]).view(1, 1, 1, 1)

    mslr_mean = torch.tensor(
        [449.9449, 308.7544, 238.3702, 220.3061]).view(1, 4, 1, 1)
    mslr_std = torch.tensor(
        [70.8778, 63.7980, 71.3171, 66.8198]).view(1, 4, 1, 1)

    model = MBA(upscale=1, pan_img_size=(height, width), pan_low_size_ratio=4, in_chans=4,
                window_size=window_size, depths=[6, 6, 6],
                embed_dim=30, num_heads=[6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', pan_mean=pan_mean, pan_std=pan_std, mslr_mean=mslr_mean, mslr_std=mslr_std)

    mslr = torch.rand((1, 4, height, width), dtype=torch.float32)
    pan = torch.rand((1, 1, height * 4, width * 4), dtype=torch.float32)

    model(pan, mslr)
    summary(model, [(1, 1, 256, 256), (1, 4, 64, 64)],
            dtypes=[torch.float32, torch.float32])
    print()
