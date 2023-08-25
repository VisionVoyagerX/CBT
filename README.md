# CBT

**A new Cross Band Transformer (CBT) architecture for pansharpening of satellite imagery**

TODO citation


# Performance on benchmark datasets


<img src="https://github.com/nickdndndn/CBT/blob/main/Images/comparison.png?raw=true" alt="alt text" width=600>


| Method      | GaoFen-2 PSNR| GaoFen-2 SSIM| WorldView-3 PSNR | WorldView-3 SSIM|
|-------------|:-------------:|:-------------:|:----------------:|:----------------:|
| PNN         | 36.3094       | 0.9350        | 31.2246          | 0.9042           |
| PanNet      | 37.9091       | 0.9533        | 33.1460          | 0.9484           |
| MSDCNN      | 38.1485       | 0.9563        | 33.5880          | 0.9552           |
| GPPNN       | 36.1538       | 0.9371        | 34.6712          | 0.9615           |
| **CBT**     | **42.5083**   | **0.9770**    | **36.6498**      | **0.9731**       |
|             |               |               |                  |                  |
| PanFormer   | 40.1886       | 0.9654        | 35.4781          | 0.9427           |
| ArbRPN      | 43.9760       | 0.9827        | 37.5401          | 0.9775           |
| **CBT_Large**| **44.8258**  | **0.9853**    | **37.7194**      | **0.9783**       |

# Performance GaoFen-2 imagery

![alt text](https://github.com/nickdndndn/CBT/blob/main/Images/visualization.png?raw=true)

# Datasets

The GaoFen-2 and WorldView-3 dataset download links can be found [here](https://github.com/liangjiandeng/PanCollection)

# TODO add Sev2Mod dataset link

# Implementation of benchmark methods used in the study

 Implementation of benchmark methods with pretrained weights on GaoFen-2 and WorldView3 datasets.
 
- [ArbRPN](https://github.com/nickdndndn/ArbRPN)
- [PanFormer](https://github.com/nickdndndn/PanFormer)
- [GPPNN](https://github.com/nickdndndn/GPPNN)
- [MSDCNN](https://github.com/nickdndndn/MSDCNN)
- [PanNet](https://github.com/nickdndndn/PanNet)
- [PNN](https://github.com/nickdndndn/PNN)

# Set Up

# Train

# Validate

`
python3 evaluate.py -c [choose config from /configs file].yaml
`

# Inference
