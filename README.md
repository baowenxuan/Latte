# Latte

[ICCV 2025] Latte: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning. Wenxuan Bao, Ruxi
Deng, Ruizhong Qiu, Tianxin Wei, Hanghang Tong, Jingrui He

# Log

Complete code is available now! 

- 2025/10/18: Dataset, core code of Latte
- 2026/05/05: Data partition, embedding caching, main function

# Prepare Data

## Download or generate data

**VLCS and TerraIncognita**

We use the dataset provided by [DomainBed](https://github.com/facebookresearch/DomainBed). 

**CIFAR-10-C and CIFAR-100-C**

Instead of using the given 10,000 samples for each dataset, we run
the [official code](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py) to generate
corrupted images for the full 60,000 samples for each dataset. The generated data can be downloaded here: 
- [CIFAR-10-C-FUll](https://huggingface.co/datasets/baowenxuan/CIFAR-10-C-Full)
- [CIFAR-100-C-Full](https://huggingface.co/datasets/baowenxuan/CIFAR-100-C-Full)

Finally, the data should be arranged as: 
```
${data_root}
│
├── domainbed
│   ├── VLCS
│   │   ├── Caltech101
│   │   │   ├── bird
│   │   │   └── ...
│   │   ├── ...
│   │   └── VOC2007
│   │       ├── bird
│   │       └── ...
│   │
│   └── terra_incognita
│       ├── location_100
│       │   ├── bird
│       │   └── ...
│       ├── ...
│       └── location_46
│           ├── bird
│           └── ...
│
└── corruption
    ├── CIFAR-10-C-Full
    │   ├── brightness.npy
    │   ├── ...
    │   ├── pixelate.npy
    │   └── labels.npy
    │
    └── CIFAR-100-C-Full
        ├── brightness.npy
        ├── ...
        ├── pixelate.npy
        └── labels.npy
```

## Cache image embeddings

For training-free TTA methods (TDA, DMN-ZS, Latte), the pre-trained model is not updated during the training. Therefore 
we can cache the image and text embeddings for more efficient experiments. To do that, run
```shell
cd ./shell
bash cache_emb.sh
```

# Run Latte

```shell
cd ./shell
bash vlcs.sh
bash terra_incognita.sh
bash cifar10c_full.sh
bash cifar100c_full.sh
```

