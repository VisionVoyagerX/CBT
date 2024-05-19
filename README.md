# CBT

**A new Cross Band Transformer (CBT) architecture for pansharpening of satellite imagery**

TODO citation


# Performance on benchmark datasets


<img src="https://github.com/nickdndndn/CBT/blob/main/Images/comparison.png?raw=true" alt="alt text" width=600>


| Method         | GaoFen-2 PSNR | GaoFen-2 SSIM | GaoFen-2 ERGAS | GaoFen-2 SAM | GaoFen-2 Q2n | WorldView-3 PSNR | WorldView-3 SSIM | WorldView-3 ERGAS | WorldView-3 SAM | WorldView-3 Q2n |
|----------------|:-------------:|:-------------:|:--------------:|:------------:|:------------:|:----------------:|:----------------:|:-----------------:|:--------------:|:---------------:|
| PNN            | 36.309        | 0.9249        | 19.4911        | 0.0495       | 0.9909       | 31.225           | 0.9042           | 70.9026           | 0.1346         | 0.9604          |
| PanNet         | 37.909        | 0.9475        | 16.2683        | 0.0412       | 0.9937       | 33.146           | 0.9440           | 56.8292           | 0.1072         | 0.9743          |
| GPPNN          | 36.154        | 0.9371        | 19.8200        | 0.0490       | 0.9908       | 34.624           | 0.9610           | 47.4321           | 0.0870         | 0.9824          |
| MDCUN          | 39.217        | 0.9561        | 13.9613        | 0.0354       | 0.9954       | 35.231           | 0.9604           | 44.1955           | 0.0823         | 0.9848          |
| BiMPan         | 40.838        | 0.9691        | 11.6978        | 0.0295       | 0.9968       | 35.342           | 0.9648           | 43.7202           | 0.0801         | 0.9853          |
| **CBT**        | **42.508**    | **0.9770**    | **9.6961**     | **0.0245**   | **0.9978**   | **36.650**       | **0.9731**       | **37.4723**       | **0.0701**     | **0.9890**      |
|                |               |               |                |              |              |                  |                  |                   |                |                 |
| PanFormer      | 40.189        | 0.9654        | 12.5479        | 0.0314       | 0.9963       | 34.956           | 0.9629           | 45.4887           | 0.0822         | 0.9844          |
| ArbRPN         | 43.976        | 0.9827        | 8.1741         | 0.0206       | 0.9984       | 37.540           | 0.9775           | 33.5207           | 0.0633         | 0.9913          |
| **CBT_Large**  | **44.826**    | **0.9853**    | **7.4158**     | **0.0186**   | **0.9987**   | **37.719**       | **0.9783**       | **32.8023**       | **0.0620**     | **0.9916**      |




# Performance GaoFen-2 imagery

![alt text](https://github.com/nickdndndn/CBT/blob/main/Images/visualization.png?raw=true)

# Datasets

The GaoFen-2 and WorldView-3 dataset download links can be found [here](https://github.com/liangjiandeng/PanCollection)
The Sev2Mod dataset can be download [here](https://zenodo.org/records/8360458)

# List of benchmark methods implemented in this study

 Implementation of benchmark methods with pretrained weights on GaoFen-2 and WorldView3 datasets.
 
- [ArbRPN](https://github.com/VisionVoyagerX/ArbRPN)
- [PanFormer](https://github.com/VisionVoyagerX/PanFormer)
- [BiMPan](https://github.com/VisionVoyagerX/BiMPan)
- [GPPNN](https://github.com/VisionVoyagerX/GPPNN)
- [MSDCNN](https://github.com/VisionVoyagerX/MDCUN)
- [PanNet](https://github.com/VisionVoyagerX/PanNet)
- [PNN](https://github.com/VisionVoyagerX/PNN)

# Project Setup

This project requires downloading datasets for training and testing purposes. Follow the steps below to set up the project:

## Step 1: Clone Repository

Clone the project repository to your local machine using the following command:

```
git clone https://github.com/VisionVoyagerX/CBT.git && cd CBT
```

## Step 2: Download Datasets and Organize

Download, untar, and store the datasets in the specified file structure and file names within the CBT root project folder.

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

**Train**

`
python3 train.py -c [choose config from /configs file].yaml #example:CBT_base_server_L_GF2.yaml
`

**Validate**

`
python3 evaluate.py -c [choose config from /configs file].yaml #example:CBT_base_server_L_GF2.yaml
`

**Inference**

`
python3 inference.py -c [choose config from /configs file].yaml #example:CBT_base_server_L_GF2.yaml
`
