# CBT

**A new Cross Band Transformer (CBT) architecture for pansharpening of satellite imagery**

TODO citation


# Performance on benchmark datasets


<img src="https://github.com/nickdndndn/CBT/blob/main/Images/comparison.png?raw=true" alt="alt text" width="75%">

![alt text](https://github.com/nickdndndn/CBT/blob/main/Images/visualization.png?raw=true)



# Performance on real world datasets

# Datasets

The GaoFen-2 and WorldView-3 dataset download links can be found [here](https://github.com/liangjiandeng/PanCollection)

Sample data (test sets) can be downloaded from [here](https://drive.google.com/file/d/1ptOImqdEM94P6Ev0Un99EjDS4CohKHO4/view?usp=sharing)

# TODO add Sev2Mod dataset link

# Benchmark methods used in the study

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
python3 evaluate.py -c CBT_base_server_T_GF2.yaml
`

# Inference
