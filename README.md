# UBK Project 

Set up experiments on deep learning models with UBK data

## Installation
```shell
pip install torch
pip install torchsampler
```

## Data preparation
Data should be processed first and put aside in `dataset` directory. The raw data and processed data is made available on request only.


## Usage
We provide MLP, AlexNet, Vgg16, Resnet18, Transformer, and Swin-transformer models for evaluation, you can conduct experiments and select 
models by `--config-file` argument shown as below. The corresponding config files are listed in `config` directory.

### Training
```shell
# cd path/to/UBKProj
python run.py --config-file config/vgg16_1d.yaml 
```

### Resume training

```shell
python run.py --config-file config/vgg16_1d.yaml TRAIN.RESUME /path/to/weight
```

### Test

```shell
python run.py --config-file config/vgg16_1d.yaml --test-only TEST.WEIGHT /path/to/weight
```

