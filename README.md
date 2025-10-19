# Latte

[ICCV 2025] Latte: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning. Wenxuan Bao, Ruxi
Deng, Ruizhong Qiu, Tianxin Wei, Hanghang Tong, Jingrui He

# Log

Code will be available soon! 

- 2025/10/18: Dataset, core code of Latte
- TODO: Data partition, embedding caching, main function

# Prepare Data

## Download Datasets

### VLCS and TerraIncognita

We use the dataset provided by [DomainBed](https://github.com/facebookresearch/DomainBed)

### CIFAR-10-C and CIFAR-100-C

Instead of using the given 10,000 samples for each dataset, we run
the [official code](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py) to generate
corrupted images for the full 50,000 samples for each dataset. 

The data should be arranged as: 
```
${data_root}
│ 
├── CIFAR-10-C-Full
│   ├── brightness.npy
│   ├── ...
│   ├── pixelate.npy
│   └── labels.npy
│ 
└── CIFAR-100-C-Full
│   ├── brightness.npy
│   ├── ...
│   ├── pixelate.npy
    └── labels.npy
```

## Cache image embeddings

For training-free TTA methods (TDA, DMN-ZS, Latte), the pre-trained model is not updated during the training. Therefore 
we can cache the image and text embeddings for more efficient experiments. To do that, run
```shell
cd ./src
CUDA_VISIBLE_DEVICES=0 python cache_emb.py \
  --dataset VLCS \
  --model 'ViT-B/16' \
  --cuda 
```

# Run Latte

```shell
cd ./src
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset VLCS \
  --model 'ViT-B/16' \
  --use_cache \
  --algo Latte \
  --cuda 
```

