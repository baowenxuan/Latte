import torch
import torch.nn.functional as F
import numpy as np
import random

import os
import argparse
import yaml

from datasets import CachedMultipleDataset, CORRUPTION_DATASETS, DOMAINBED_DATASETS
from utils import stratified_partition_domainbed, stratified_partition_corruption, get_domain_stat

from methods import BaseTTAServer, BaseClient
from methods import LatteServer, LatteClient


def main(args):
    # Load Dataset
    datasets = CachedMultipleDataset(args.load_cache_from)

    # Dataset partition
    if args.dataset in DOMAINBED_DATASETS:
        client_datasets = stratified_partition_domainbed(datasets, k=args.num_clients_per_domain, seed=args.seed)
    elif args.dataset in CORRUPTION_DATASETS:
        client_datasets = stratified_partition_corruption(datasets, k=args.num_clients_per_domain, seed=args.seed)
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

    args.num_clients = len(client_datasets)

    clip_weights = torch.stack(datasets.text_embeddings, dim=0).to(device=args.device, dtype=args.dtype)
    clip_weights = F.normalize(clip_weights.mean(dim=1), dim=1)  # average across multiple templates

    if args.algo == 'no_adapt':
        server = BaseTTAServer(datasets=client_datasets, clip_weights=clip_weights, args=args,
                               client_class=BaseClient)

    elif args.algo == 'latte':
        server = LatteServer(datasets=client_datasets, clip_weights=clip_weights, args=args,
                             client_class=LatteClient)

    else:
        raise NotImplementedError('Unknown algorithm: {}'.format(args.algo))

    acc, _, _, client_stats = server.evaluate()

    print(f"Acc: {acc:.6f}")

    print(client_stats)

    domain_stats = get_domain_stat(client_stats, client_datasets, datasets.environments)

    print(domain_stats)

    return acc, client_stats, domain_stats


def args_parser():
    parser = argparse.ArgumentParser()

    # Datasets

    parser.add_argument('--cache_root', type=str, default='../cache')

    parser.add_argument('--dataset', type=str, default='VLCS')

    parser.add_argument('--num_clients_per_domain', type=int, default=1,
                        help='number of client for each domain')

    # Model

    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help='model name, a version of CLIP listed by clip.available_models()')

    # Algorithm

    parser.add_argument('--algo', type=str, default='clip',
                        help='algorithm name')

    parser.add_argument('--config_root', type=str, default='../configs')

    # For CTTA only

    parser.add_argument('--part_rate', type=float, default=1.0)

    parser.add_argument('--sync_freq', type=int, default=1,
                        help='frequency of synchronization, 5 means synchronize every 5 rounds')

    # Setting

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataloader')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads')

    args = parser.parse_args()

    args.is_corruption_dataset = (args.dataset in CORRUPTION_DATASETS)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    args.dtype = torch.half if torch.cuda.is_available() and args.cuda else torch.float
    # args.dtype_name = 'half' if torch.cuda.is_available() and args.cuda else 'float'
    # args.dtype = torch.float
    args.dtype_name = 'half'
    args.model_name = args.model.replace('/', '-')

    args.load_cache_from = os.path.expanduser(args.cache_root)
    args.load_cache_from = os.path.join(args.load_cache_from, args.dataset, args.model.replace('/', '-'))

    try:
        filepath = os.path.expanduser(os.path.join(args.config_root, args.dataset, args.algo + '.yaml'))
        with open(filepath, 'r') as f:
            args.config = yaml.safe_load(f)
    except:
        args.config = {}

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    torch.set_num_threads(args.num_threads)

    setup_seed(args.seed)
    acc, client_stats, domain_stats = main(args)
