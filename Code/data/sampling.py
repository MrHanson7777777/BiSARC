#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data sampling module —— IID / Non-IID data partitioning"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """Wrapper class to split dataset by indices"""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        if isinstance(data, torch.Tensor):
            data = data.clone().detach()
        else:
            data = torch.tensor(data)
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
        return data, label



def iid_split(dataset, num_users):
    """Perform IID uniform partitioning on any dataset"""
    num_items = len(dataset) // num_users
    all_idxs = list(range(len(dataset)))
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_shard_split(dataset, num_users, num_shards=200, num_imgs=None):
    """Non-IID partitioning by splitting shards after sorting by labels"""
    if num_imgs is None:
        num_imgs = len(dataset) // num_shards
    total = num_shards * num_imgs

    labels = np.array(dataset.targets)[:total]
    idxs = np.arange(total)
    idxs = idxs[labels.argsort()]

    shard_ids = list(range(num_shards))
    np.random.shuffle(shard_ids)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        assigned = shard_ids[i::num_users]
        for s in assigned:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[s * num_imgs:(s + 1) * num_imgs]))
    return dict_users


def noniid_dirichlet_split(dataset, num_users, alpha=0.5):
    """Use Dirichlet distribution for Non-IID data partitioning"""
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    class_idxs = [np.where(labels == c)[0].tolist() for c in range(num_classes)]

    total_samples_per_user = len(dataset) // num_users

    for i in range(num_users):
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        client_data = []
        for c in range(num_classes):
            n_take = int(proportions[c] * total_samples_per_user)
            n_take = min(n_take, len(class_idxs[c]))
            if n_take > 0:
                selected = np.random.choice(class_idxs[c], n_take, replace=False)
                client_data.extend(selected)
                class_idxs[c] = list(set(class_idxs[c]) - set(selected))
        dict_users[i] = np.array(client_data, dtype='int64')

    return dict_users
