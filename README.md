```

==================================================================================================================================
Layer (type:depth-idx)                                                           Output Shape              Param #
==================================================================================================================================
CrossFormer                                                                      [1, 4, 256, 256]          --
├─Shallow_Feature_Extractor: 1-1                                                 [1, 48, 256, 256]         --
│    └─Conv2d: 2-1                                                               [1, 48, 256, 256]         480
├─Shallow_Feature_Extractor: 1-2                                                 [1, 48, 256, 256]         --
│    └─Image_Reconstruction: 2-2                                                 [1, 48, 256, 256]         --
│    │    └─Sequential: 3-1                                                      [1, 48, 64, 64]           1,776
│    │    └─Upsample: 3-2                                                        [1, 48, 256, 256]         166,272
│    │    └─Conv2d: 3-3                                                          [1, 48, 256, 256]         20,784
├─Deep_Feature_Extractor: 1-3                                                    [1, 48, 256, 256]         96
│    └─PatchEmbed: 2-3                                                           [1, 65536, 48]            --
│    │    └─LayerNorm: 3-4                                                       [1, 65536, 48]            96
│    └─PatchEmbed: 2-4                                                           [1, 65536, 48]            --
│    │    └─LayerNorm: 3-5                                                       [1, 65536, 48]            96
│    └─Dropout: 2-5                                                              [1, 65536, 48]            --
│    └─Dropout: 2-6                                                              [1, 65536, 48]            --
│    └─ModuleList: 2-7                                                           --                        --
│    │    └─RHAG: 3-6                                                            [1, 65536, 48]            517,980
│    │    └─RHAG: 3-7                                                            [1, 65536, 48]            517,980
│    └─LayerNorm: 2-8                                                            [1, 65536, 48]            96
│    └─LayerNorm: 2-9                                                            [1, 65536, 48]            (recursive)
│    └─PatchUnEmbed: 2-10                                                        [1, 48, 256, 256]         --
│    └─PatchUnEmbed: 2-11                                                        [1, 48, 256, 256]         --
│    └─Conv2d: 2-12                                                              [1, 48, 256, 256]         20,784
│    └─Conv2d: 2-13                                                              [1, 48, 256, 256]         20,784
├─Image_Reconstruction: 1-4                                                      [1, 4, 256, 256]          --
│    └─Sequential: 2-14                                                          [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-8                                                          [1, 64, 256, 256]         27,712
│    │    └─LeakyReLU: 3-9                                                       [1, 64, 256, 256]         --
│    └─Upsample: 2-15                                                            [1, 64, 256, 256]         --
│    └─Conv2d: 2-16                                                              [1, 4, 256, 256]          2,308
==================================================================================================================================
Total params: 1,297,244
Trainable params: 1,297,244
Non-trainable params: 0
Total mult-adds (G): 16.12
==================================================================================================================================
Input size (MB): 0.33
Forward/backward pass size (MB): 9354.88
Params size (MB): 4.00
Estimated Total Size (MB): 9359.21
==================================================================================================================================

```
