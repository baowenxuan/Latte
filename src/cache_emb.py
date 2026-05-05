import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
import os
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

from datasets import get_dataset_class
from datasets.augmix import AugMixAugmenter
from utils import pickle_save


@torch.no_grad()
def encode_text(clip_model, class_names, templates):
    all_texts = [[t.format(c.replace('_', ' ')) for t in templates] for c in class_names]

    text_features = []

    device = next(clip_model.parameters()).device

    for texts in tqdm(all_texts):
        texts = clip.tokenize(texts).to(device)
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings = F.normalize(class_embeddings, dim=1).cpu()
        text_features.append(class_embeddings)

    return text_features


@torch.no_grad()
def encode_image(clip_model, dataset):
    image_features, labels = [], []

    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype

    for image, label in tqdm(dataset):
        if isinstance(image, list):  # list of images
            image = torch.stack(image, dim=0).to(device=device, dtype=dtype)
        else:  # single image
            image = image.unsqueeze(0).to(device=device, dtype=dtype)

        image_feature = clip_model.encode_image(image)  # batch_size * feat_dim
        image_feature = F.normalize(image_feature, dim=1).cpu()

        image_features.append(image_feature)
        labels.append(label)

    # Save it as a large tensor
    image_features = torch.stack(image_features, dim=0)  # num_sample * batch_size * feat_dim
    labels = torch.LongTensor(labels)

    return image_features, labels


def main(args):
    print(args)
    clip_model, clip_preprocess = clip.load(args.model, device=args.device)

    clip_model.eval()
    clip_model.requires_grad_(False)

    # Image embedding

    if args.use_augmix:
        # Use Augmix Augmentation
        base_transform = transforms.Compose(clip_preprocess.transforms[:3])  # resize, centercrop, to_rgb
        preprocess = transforms.Compose(clip_preprocess.transforms[3:])  # to_tensor, normalize
        transform = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    else:
        transform = clip_preprocess

    datasets = get_dataset_class(args.dataset)(root=args.data_root, transform=transform)

    cache = {}

    for i in range(len(datasets)):
        env_name = datasets.environments[i]
        dataset = datasets[i]
        path = args.save_img_emb_to.format(env_name)

        image_features, labels = encode_image(clip_model, dataset)
        pickle_save((image_features, labels), path)
        tqdm.write("Saved image features to {}".format(path))

    # Text embedding

    if args.use_tip:
        templates = [
            "itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}.",
        ]
    else:
        templates = [
            "a photo of a {}."
        ]

    text_features = encode_text(clip_model, datasets.classes, templates)
    pickle_save(text_features, args.save_text_emb_to)
    tqdm.write("Saved text features to {}".format(args.save_text_emb_to))

    # Meta info
    meta_info = {
        'args': args,
        'environments': datasets.environments,
        'classes': datasets.classes,
    }

    pickle_save(meta_info, args.save_meta_info_to)
    tqdm.write("Saved meta info to {}".format(args.save_meta_info_to))




def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10CFull',
                        help='dataset name')

    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help='model name, a version of CLIP listed by clip.available_models()')

    parser.add_argument('--data_root', type=str, default='~/data',
                        help='dataset root path')

    parser.add_argument('--cache_root', type=str, default='../cache',
                        help='cache root path, cached embedding will be saved in this folder')

    parser.add_argument('--use_augmix', action='store_true', default=False,
                        help='whether to use augmix to generate image embeddings')

    parser.add_argument('--use_tip', action='store_true', default=False,
                        help='whether to use seven templates')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataloader')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads')

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda')

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    args.save_to = os.path.expanduser(args.cache_root)
    args.save_to = os.path.join(args.save_to, args.dataset, args.model.replace('/', '-'))
    args.save_text_emb_to = os.path.join(args.save_to, f"text_{'tip' if args.use_tip else 'default'}.pkl")
    args.save_img_emb_to = os.path.join(args.save_to, f"img_{'augmix' if args.use_augmix else 'default'}_{{}}.pkl")
    args.save_meta_info_to = os.path.join(args.save_to, f"meta_info.pkl")

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
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    main(args)
