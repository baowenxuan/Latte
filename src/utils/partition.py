import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold


def stratitied_k_fold(labels, k, seed=0):
    if k > 1:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = [fold for _, fold in skf.split(np.arange(len(labels)), labels)]
    else:
        folds = [np.arange(len(labels)), ]  # only one fold

    # shuffle within each fold
    np.random.seed(seed)
    for fold in folds:
        np.random.shuffle(fold)

    return folds


def stratified_partition_domainbed(datasets, k, seed=0):
    client_datasets = {}

    for i, dataset in enumerate(datasets):

        environment = datasets.environments[i]

        if isinstance(dataset, TensorDataset):
            labels = dataset.tensors[1]
        elif isinstance(dataset, ImageFolder):
            labels = dataset.targets
        else:  # this will take a long time
            labels = torch.LongTensor([Y for (X, Y) in dataset])

        folds = stratitied_k_fold(labels, k, seed)

        client_datasets.update({f"{environment}_{j}": Subset(dataset, fold) for j, fold in enumerate(folds)})

    return client_datasets


def stratified_partition_corruption(datasets, k, seed=0):
    """
    For corruption datasets, datasets[domain_i] and datasets[domain_j] are un-shuffled variants of the same dataset of
    image. Therefore, we use a different partition function to make sure their sample_idx do not overlap
    :param datasets:
    :param k:
    :param seed:
    :return:
    """
    client_datasets = {}
    num_domains = len(datasets)

    dataset = datasets[0]

    # instead of partition each domain to k fold, we partition the whole

    if isinstance(dataset, TensorDataset):
        labels = dataset.tensors[1]
    elif isinstance(dataset, ImageFolder):
        labels = dataset.targets
    else:  # this will take a long time
        labels = torch.LongTensor([Y for (X, Y) in dataset])

    folds = stratitied_k_fold(labels, num_domains * k, seed)

    np.random.seed(0)
    np.random.shuffle(folds)  # necessary here because fold in folds are sorted with their smallest idx

    for i, dataset in enumerate(datasets):

        environment = datasets.environments[i]

        for j in range(k):
            fold = folds[i * k + j]
            client_datasets[f"{environment}_{j}"] = Subset(dataset, fold)

    return client_datasets
