import torch
import torch.nn.functional as F
import numpy as np
from bisect import bisect_left
import math

# from torchprofile import profile_macs

from .Base import BaseCTTAServer, BaseClient
from .utils import get_clip_logits, get_entropy_batch


class LatteClient(BaseClient):

    def __init__(self, dataset, clip_weights, args):

        super(LatteClient, self).__init__(dataset, clip_weights, args)

        self.config = args.config

        local_cfg, global_cfg = args.config['local'], args.config['global']

        self.local_enabled, self.global_enabled = local_cfg['enabled'], global_cfg['enabled']

        self.local_params = {k: local_cfg[k] for k in
                             ['shot_capacity', 'alpha', 'beta', 'gamma']}

        self.global_params = {k: global_cfg[k] for k in
                              ['shot_capacity', 'prototype', 'gamma']}

        # shot capacity should not exceed number of clients - 1, since each client only upload one shot
        self.global_params['shot_capacity'] = min(self.global_params['shot_capacity'], args.num_clients - 1)

        # Local cache and entropy

        # Instead of keep sorting the vectors, which is not efficient, we sort their indices in the matrix, according to
        # their entropy

        self.local_ent = [[] for class_idx in range(self.num_class)]  # list of lists of [entropy, idx]

        self.local_cache = torch.zeros(self.num_class, self.local_params['shot_capacity'], self.feat_dim,
                                       device=self.device, dtype=self.dtype)

        self.local_cache_ent = torch.ones(self.num_class, self.local_params['shot_capacity'],
                                          device=self.device, dtype=self.dtype)

        # External cache and entropy

        self.external_cache = torch.zeros(self.num_class, self.global_params['shot_capacity'], self.feat_dim,
                                          device=self.device, dtype=self.dtype)

        self.external_cache_ent = torch.ones(self.num_class, self.global_params['shot_capacity'],
                                             device=self.device, dtype=self.dtype)

        self.max_ent = math.log(self.num_class)  # max possible entropy, given c classes

        self.merged_cache_last = None
        self.merged_cache_ent_last = None

    @torch.no_grad()
    def update_cache(self, cache, cache_ent, ent_lists, pred, features_loss, shot_capacity):
        """
        Update the local cache with new image embedding
        """

        feature, new_ent = features_loss[:2]

        if len(ent_lists[pred]) >= shot_capacity:

            old_ent, old_idx = ent_lists[pred][-1]

            if new_ent < old_ent:
                # replace feature in the cache
                cache[pred][old_idx] = feature
                cache_ent[pred][old_idx] = new_ent / self.max_ent

                # maintain the entropy queue
                ent_lists[pred].pop()  # delete the old one
                new_item = (new_ent, old_idx)
                insert_idx = bisect_left(ent_lists[pred], new_item)
                ent_lists[pred].insert(insert_idx, new_item)  # insert the new one, make sure still sorted

        else:

            idx = len(ent_lists[pred])

            # add a new feature into the cache
            cache[pred][idx] = feature
            cache_ent[pred][idx] = new_ent / self.max_ent

            # maintain the entropy queue
            new_item = (new_ent, idx)
            insert_idx = bisect_left(ent_lists[pred], new_item)
            ent_lists[pred].insert(insert_idx, new_item)  # insert the new one, make sure still sorted

    @torch.no_grad()
    def upload_cache(self, prototype='min_ent', gamma=0):
        """
        Upload prototypes (and their entropy) to the server
        :param prototype: how to construct class prototype
        :return: local_cache_to_upload: torch.Tensor, num_class * feat_dim
        :return: ents: list of floats
        """

        if prototype == 'mean':  # equivalent to exp_weighted with gamma = 0
            prototype_to_upload = F.normalize(self.local_cache.mean(dim=1), dim=1)

            ents = get_entropy_batch(prototype_to_upload, self.clip_weights) / self.max_ent

        elif prototype == 'exp_weighted':  # the strategy used in paper

            weights = (- gamma * self.local_cache_ent).exp().unsqueeze(2)  # num_class * shot_capacity * 1
            prototype_to_upload = F.normalize((self.local_cache * weights).mean(dim=1), dim=1)
            ents = get_entropy_batch(prototype_to_upload, self.clip_weights) / self.max_ent


        elif prototype == 'min_ent':  # upload the sample with min entropy, equivalent to gamma -> infty
            min_ent_idx = torch.LongTensor(
                [ent_idx_list[0][1] if ent_idx_list else 0 for ent_idx_list in self.local_ent])

            class_idx = torch.arange(self.num_class)
            prototype_to_upload = self.local_cache[class_idx, min_ent_idx]  # num_class * feat_dim
            ents = self.local_cache_ent[class_idx, min_ent_idx]  # reuse the computed entropy

        else:
            raise NotImplementedError

        return prototype_to_upload, ents

    @torch.no_grad()
    def download_cache(self, external_cache, external_cache_ent):
        """
        Download the global cache from the server
        :param personalized_cache:
        :return:
        """

        self.external_cache = external_cache
        self.external_cache_ent = external_cache_ent

    @torch.no_grad()
    def compute_cache_logits(self, image_feature, cache, cache_ent, alpha, beta, gamma):

        affinity = cache @ image_feature.T  # num_class * shot_capacity * batch_size
        attn = (beta * (affinity - 1)).exp()  # num_class * shot_capacity * batch_size

        attn.mul_((- gamma * cache_ent.unsqueeze(2)).exp())  # assign smaller weight to larger entropy

        adaptive_cls_weight = (cache.unsqueeze(3) * attn.unsqueeze(2)).sum(dim=1)  # num_class * feat_dim * batch_size

        adaptive_cls_weight = adaptive_cls_weight.permute(2, 0, 1)  # batch_size * num_class * feat_dim
        is_empty = adaptive_cls_weight.norm(dim=2) <= 1e-3
        adaptive_cls_weight = F.normalize(adaptive_cls_weight, dim=2)
        adaptive_cls_weight[is_empty] = 0.0

        cache_logits = 100.0 * (image_feature.unsqueeze(1) * adaptive_cls_weight).sum(dim=2)

        # print(cache_logits.shape)

        return alpha * cache_logits

    @torch.no_grad()
    def predict(self, image_feature):

        # Step 1. Get initial prediction
        clip_logits, pred, proba, entropy, image_feature = get_clip_logits(image_feature, self.clip_weights,
                                                                           return_features=True)

        # Step 2. Update local memory

        self.update_cache(self.local_cache, self.local_cache_ent, self.local_ent, pred, [image_feature, entropy],
                          self.local_params['shot_capacity'])

        # Step 3. Get prediction

        final_logits = clip_logits.clone()

        if self.global_enabled and self.local_enabled:
            cache = torch.cat([self.local_cache, self.external_cache], dim=1)
            cache_ent = torch.cat([self.local_cache_ent, self.external_cache_ent], dim=1)

            selected = cache_ent.argsort(dim=1, descending=False)[:, :self.local_params['shot_capacity']]

            cache = torch.gather(cache, 1, selected.unsqueeze(-1).expand(-1, -1, self.feat_dim))
            cache_ent = torch.gather(cache_ent, 1, selected)

            self.merged_cache_last = cache
            self.merged_cache_ent_last = cache_ent

        elif self.local_enabled:  # only use local cache
            cache = self.local_cache
            cache_ent = self.local_cache_ent

        elif self.global_enabled:  # only use external cache
            cache = self.external_cache
            cache_ent = self.external_cache_ent

        else:
            final_pred = final_logits.argmax(dim=1).item()
            return final_pred

        final_logits += self.compute_cache_logits(image_feature, cache, cache_ent, self.local_params['alpha'],
                                                  self.local_params['beta'], self.local_params['gamma'])

        final_pred = final_logits.argmax(dim=1).item()

        return final_pred


class LatteServer(BaseCTTAServer):

    def __init__(self, datasets, clip_weights, args, client_class=LatteClient):
        super(LatteServer, self).__init__(datasets, clip_weights, args, client_class)

        global_cfg = args.config['global']

        self.global_params = {k: global_cfg[k] for k in
                              ['shot_capacity', 'prototype', 'gamma']}

        self.global_params['shot_capacity'] = min(self.global_params['shot_capacity'], args.num_clients - 1)

        self.global_cache = torch.zeros(self.num_class, self.num_clients, self.feat_dim,
                                        device=self.device, dtype=self.dtype)

        self.global_cache_ent = torch.ones(self.num_class, self.num_clients,
                                           device=self.device, dtype=self.dtype)

        self.collab_matrix = []

    def syncronize(self):

        # select a subset of clients
        selected_client_idxs = sorted(list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))

        self.collab_matrix.append(np.zeros((self.num_clients, self.num_clients)))

        # collect prototypes from each client, and update the global cache

        for client_idx in selected_client_idxs:
            cid = self.idx2cid[client_idx]
            client = self.clients[cid]
            prototypes, ents = client.upload_cache(prototype=self.global_params['prototype'],
                                                   gamma=self.global_params['gamma'])

            self.global_cache[:, client_idx, :] = prototypes
            self.global_cache_ent[:, client_idx] = ents

        # compute personalized cache

        for client_idx in selected_client_idxs:
            cid = self.idx2cid[client_idx]
            client = self.clients[cid]

            personal_cache, personal_cache_ent, selected_idx = self.subset_selection(self.global_cache,
                                                                                     self.global_cache_ent,
                                                                                     client_idx, self.global_params[
                                                                                         'shot_capacity'])

            for selected_idx_per_class in selected_idx:
                self.collab_matrix[-1][client_idx, selected_idx_per_class] += 1

            client.download_cache(personal_cache, personal_cache_ent)

    def subset_selection(self, global_cache, global_cache_ent, query_idx, topk):
        """
        For each class, find the topk vector that is most similar to the query cache
        :param global_cache: num_class * n_shot * feat_dim
        :param query_idx:
        :return:
        """

        query_cache = global_cache[:, query_idx, :]

        similarity = (global_cache * query_cache.unsqueeze(1)).sum(dim=2)  # num_class * n_clients

        selected_idx = torch.sort(similarity, dim=1, descending=True).indices[:,
                       1: (topk + 1)]  # exclude the query itself

        personal_cache_ent = global_cache_ent.gather(1, selected_idx)

        selected = selected_idx.unsqueeze(-1).expand(-1, -1, global_cache.shape[2])

        personal_cache = global_cache.gather(1, selected)

        return personal_cache, personal_cache_ent, selected_idx.cpu().numpy()
