Code and Dataset of paper "Decoupled Cross-Modal Alignment Network for Text-RGBT Person Retrieval and A High-Quality Benchmark"

# Datasets
RGBT-PEDE is the first publicly available dataset designed for text-based RGBT person retrieval. It contains 1,822 unique identities, multi-modal images (RGB and thermal), and fine-grained textual descriptions. The dataset covers a wide range of scenes across daytime and nighttime, featuring various challenges such as occlusion and illumination changes.

![image](https://github.com/Yifei-AHU/RGBT-PEDE/blob/main/images/Fig10.png?raw=true))

# Methods
![image](https://github.com/Yifei-AHU/RGBT-PEDE/blob/main/images/Fig2.png?raw=true)

# Prepare Datasets

You can download the RGBT-PEDES dataset from Baidu Netdisk

## Baidu Netdisk
Link: https://pan.baidu.com/s/1cbUO9vUeJa1svDMmoCvOAg 

Password: xixz

# Training
Our code borrows partially from IRRA.
we use single RTX4090 24G GPU for training and evaluation.

# Testing
'''
python test.py --config_file 'path/to/model_dir/configs.yaml'
'''
