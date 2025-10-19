import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from collections import OrderedDict

from .utils import get_clip_logits


class BaseClient:

    def __init__(self, dataset, clip_weights, args):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.curr_idx = 0
        self.clip_weights = clip_weights
        self.num_class, self.feat_dim = clip_weights.shape

        self.dtype = clip_weights.dtype
        self.device = clip_weights.device

        self.num_correct = 0

        self.args = args

    @torch.no_grad()
    def predict(self, image_feature):
        """
        Given the image_feature, how to get prediction and update the "state" of the classifier.
        This is the function reloaded by other algorithms.
        :param image_feature:
        :return:
        """
        # image_feature is already (1) normalized and (2) converted to the correct device and dtype

        clip_logits, pred, _, _ = get_clip_logits(image_feature, self.clip_weights,
                                                  normalize=False)  # already normalized

        return pred

    def evaluate_one(self):
        """
        Wrapper of "predict", since dataloading and metrics computing can be reused.
        :return:
        """
        if self.curr_idx >= self.num_samples:  # already finish
            return 0, 0  # 0 correct over 0 sample

        image_feature, label = self.dataset[self.curr_idx]
        self.curr_idx += 1

        image_feature = image_feature.to(device=self.device, dtype=self.dtype)

        image_feature = F.normalize(image_feature, dim=1)

        pred = self.predict(image_feature)

        correct = int(pred == label.item())

        self.num_correct += correct

        return correct, 1  # ? correct over 1 sample

    def is_done(self):
        return self.curr_idx >= self.num_samples


class BaseTTAServer:

    def __init__(self, datasets, clip_weights, args, client_class=BaseClient):
        """

        :param datasets: dictionary of client_id : dataset. Make sure the dataset is permuted before feeding.
        :param args:
        """

        self.num_class, self.feat_dim = clip_weights.shape

        self.clients = OrderedDict(
            [(cid, client_class(dataset, clip_weights, args)) for (cid, dataset) in datasets.items()])

        self.client_ids = list(datasets.keys())

        self.num_clients = len(datasets)

        # conversion between client id and client index
        self.cid2idx = {cid: idx for idx, cid in enumerate(self.clients)}
        self.idx2cid = {idx: cid for cid, idx in self.cid2idx.items()}

        self.dtype = clip_weights.dtype
        self.device = clip_weights.device

    def evaluate(self):
        total_correct, total_num_samples = 0, 0

        for client in tqdm(self.clients.values()):

            client_correct, client_num_samples = 0, 0

            finished = False

            for i in tqdm(range(client.num_samples)):
                correct, total = client.evaluate_one()
                client_correct += correct
                client_num_samples += total
                finished = (total == 0)

            total_correct += client_correct
            total_num_samples += client_num_samples

        stats = {cid: client.num_correct / client.num_samples for (cid, client) in self.clients.items()}

        return total_correct / total_num_samples, total_correct, total_num_samples, stats


class BaseCTTAServer(BaseTTAServer):

    def __init__(self, datasets, clip_weights, args, client_class=BaseClient):
        """

        :param datasets: dictionary of client_id : dataset. Make sure the dataset is permuted before feeding.
        :param args:
        """

        super(BaseCTTAServer, self).__init__(datasets, clip_weights, args, client_class)

        self.cohort_size = int(self.num_clients * args.part_rate)  # number of clients participate in each iter

        self.num_rounds = sum(client.num_samples for client in self.clients.values())  # Just an upper bound

        self.sync_freq = args.sync_freq

    def evaluate(self):

        total_correct, total_num_samples = 0, 0

        for rnd in tqdm(range(1, self.num_rounds + 1)):  # TODO: partial participant?

            # select a subset of clients and evaluate
            np.random.shuffle(self.client_ids)
            selected_idx = sorted(list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))

            rnd_correct, rnd_num_samples = 0, 0

            for idx in selected_idx:
                client = self.clients[self.client_ids[idx]]  # idx -> client_id -> client
                correct, num_samples = client.evaluate_one()
                rnd_correct += correct
                rnd_num_samples += num_samples

            total_correct += rnd_correct
            total_num_samples += rnd_num_samples

            # break the loop if no more testing samples
            if all(client.is_done() for client in self.clients.values()):
                break

            # placeholder for synchronization (no synchronization by default)
            if rnd % self.sync_freq == 0:
                # print('Synchronizing ... ')
                self.syncronize()

        # accuracy for each individual clients
        stats = {cid: client.num_correct / client.num_samples for (cid, client) in self.clients.items()}

        return total_correct / total_num_samples, total_correct, total_num_samples, stats

    def syncronize(self):
        # By default, do nothing
        pass
