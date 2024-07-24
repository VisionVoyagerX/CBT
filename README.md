# CBT

**A new Cross Band Transformer (CBT) and a Wavelet Cross Band Transformer (Wav-CBT) architecture for pansharpening of satellite imagery**


# Performance on benchmark datasets


## Model performance in terms of PSNR (dB), SSIM, ERGAS, and SAM for CBT and various benchmark methods

Methods are classified into small-scale (top) and large-scale models (bottom), highlighting the **best**, _second best_, and *third best* scores.

| Method                        | PSNR↑    | SSIM↑    | ERGAS↓   | SAM↓     | PSNR↑    | SSIM↑    | ERGAS↓   | SAM↓     |
|-------------------------------|----------|----------|----------|----------|----------|----------|----------|----------|
|                               |          |          | GaoFen-2 |          |          |          |WorldView-3|         |
| **Small-scale models**        |          |          |          |          |          |          |          |          |
| PNN                           | 36.309   | 0.9249   | 1.3682   | 0.0241   | 31.225   | 0.9042   | 4.4628   | 0.0815   |
| PanNet                        | 37.909   | 0.9475   | 1.1374   | 0.0197   | 33.146   | 0.9440   | 3.5800   | 0.0702   |
| GPPNN                         | 36.154   | 0.9371   | 1.3683   | 0.0230   | 34.624   | 0.9610   | 2.8693   | 0.0678   |
| MSDCNN                        | 38.136   | 0.9497   | 1.1202   | 0.0193   | 33.560   | 0.9493   | 3.2983   | 0.0696   |
| MDCUN                         | 39.217   | 0.9561   | 0.9808   | 0.0167   | 35.231   | 0.9604   | 2.7164   | 0.0613   |
| BiMPan                        | _39.638_ | _0.9616_ | _0.9363_ | _0.0173_ | _35.342_ | _0.9648_ | _2.6146_ | _0.0623_ |
| **CBT (ours)**                | **42.508** | **0.9770** | **0.6731** | **0.0128** | **36.650** | **0.9731** | **2.2975** | **0.0510** |
| **Wav-CBT (ours)**            | **42.661** | **0.9776** | **0.6591** | **0.0125** | **36.427** | **0.9715** | **2.3513** | **0.0526** |
| **Large-scale models**        |          |          |          |          |          |          |          |          |
| PanFormer                     | 40.189   | 0.9654   | 0.8709   | 0.0161   | 34.956   | 0.9629   | 2.8153   | 0.0613   |
| ArbRPN                        | _43.976_ | _0.9827_ | _0.5632_ | _0.0110_ | _37.540_ | _0.9775_ | _2.0356_ | _0.0481_ |
| **CBT_Large (ours)**  | **44.826** | **0.9853** | **0.5075** | **0.0099** | **37.719** | **0.9783** | **1.9977** | **0.0472** |
| **Wav-CBT_Large(ours)** | **44.980** | **0.9857** | **0.4980** | **0.0098** | _37.033_ | _0.9757_ | _2.1620_ | _0.0485_ |


# Performance GaoFen-2 imagery

![alt text](https://github.com/nickdndndn/CBT/blob/main/Images/visualization.png?raw=true)

# Datasets

The GaoFen-2 and WorldView-3 dataset download links can be found [here](https://github.com/liangjiandeng/PanCollection)
The Sev2Mod dataset can be download [here](https://zenodo.org/records/8360458)

# List of benchmark methods implemented in this study

 Implementation of benchmark methods with pretrained weights on GaoFen-2 and WorldView3 datasets.
 
- [PNN](https://github.com/VisionVoyagerX/PNN)
- [PanNet](https://github.com/VisionVoyagerX/PanNet)
- [GPPNN](https://github.com/VisionVoyagerX/GPPNN)
- [MSDCNN](https://github.com/VisionVoyagerX/MDCUN)
- [BiMPan](https://github.com/VisionVoyagerX/BiMPan)
- [PanFormer](https://github.com/VisionVoyagerX/PanFormer)
- [ArbRPN](https://github.com/VisionVoyagerX/ArbRPN)

# Project Setup

This project requires downloading datasets for training and testing purposes. Follow the steps below to set up the project:

## Step 1: Clone Repository

Clone the project repository to your local machine using the following command:

```
git clone https://github.com/VisionVoyagerX/CBT.git && cd CBT
```

## Step 2: Download Datasets and Organize

Download and extract the datasets, then organize them according to the specified file structure below. Ensure the file is placed in the same directory as the CBT project.

- GF2
    - train
    - val
    - test
- WV3
    - train
    - val
    - test
- SEV2MOD
    - train
    - val
    - test

Datasets and their respective URLs are listed below. Download, untar, and store them in the specified file structure within the CBT root project folder.

- GF2
    - Train-Val: [GF2 Train-Val](https://drive.google.com/drive/folders/1gNV7BlGy06ee0BqgxBfFMNnfzGrPTA9K)
    - Test: [GF2 Test](https://drive.google.com/drive/folders/1g4f2NElV7By2gWhCavrDaglzCxiDT6CP)
- WV3
    - Train-Test: [WV3 Train-Test](https://drive.google.com/drive/folders/1CHs49xius3zH3PIrAxAkbNfKEy82_fMb)
    - Test: [WV3 Test](https://drive.google.com/drive/folders/1EYjaAxTheNPvukvifKXMq8m_dJ-8qz8G)
- SEV2MOD
    - Train-Val-Test: [SEV2MOD Train-Val-Test](https://zenodo.org/records/8360458)

## Step 3: Train (optional)

`
python3 train.py -c [choose config from /configs file].yaml #example:CBT_base_server_L_GF2.yaml
`

## Step 4: Inference

`
python3 inference.py -c [choose config from /configs file].yaml #example:CBT_base_server_L_GF2.yaml
`
