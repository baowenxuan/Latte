import torch
from torch.utils.data import TensorDataset
import os

from utils import pickle_load


class CachedMultipleDataset:
    def __init__(self, root):

        # Load meta info first
        self.meta_info = pickle_load(os.path.join(root, "meta_info.pkl"))
        self.environments = self.meta_info["environments"]
        self.classes = self.meta_info["classes"]

        img_emb_path = self.meta_info['args'].save_img_emb_to
        text_emb_path = self.meta_info['args'].save_text_emb_to

        # Load image embeddings
        self.datasets = []
        for env_name in self.environments:
            image_features, labels = pickle_load(img_emb_path.format(env_name))
            self.datasets.append(TensorDataset(image_features, labels))

        # Load text embeddings
        self.text_embeddings = pickle_load(text_emb_path)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]



