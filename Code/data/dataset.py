#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset loading module —— Unified loading of MNIST / CIFAR-10 / CIFAR-100 / FEMNIST / Shakespeare"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from .sampling import iid_split, noniid_shard_split, noniid_dirichlet_split


def get_dataset(args):
    """
    Load train and test datasets based on args, and perform client data partitioning.

    Returns:
        train_dataset, test_dataset, user_groups(dict)
    """
    if args.dataset == 'mnist':
        return _load_mnist(args)
    elif args.dataset == 'cifar':
        return _load_cifar10(args)
    elif args.dataset == 'cifar100':
        return _load_cifar100(args)
    elif args.dataset == 'femnist':
        return _load_femnist(args)
    elif args.dataset == 'shakespeare':
        return _load_shakespeare(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}, available: mnist, cifar, cifar100, femnist, shakespeare")


# MNIST

def _load_mnist(args):
    data_dir = './data/mnist/'
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_tf = train_tf  # MNIST test set uses the same normalization

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)

    user_groups = _split(train_ds, args)
    return train_ds, test_ds, user_groups


# CIFAR-10

def _load_cifar10(args):
    data_dir = './data/cifar/'
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    user_groups = _split(train_ds, args)
    return train_ds, test_ds, user_groups


# CIFAR-100

def _load_cifar100(args):
    data_dir = './data/cifar100/'
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_tf)

    user_groups = _split(train_ds, args)
    return train_ds, test_ds, user_groups


# Unified partition entry

def _split(dataset, args):
    if args.iid:
        return iid_split(dataset, args.num_users)
    else:
        alpha = getattr(args, 'alpha', None)
        if alpha is not None:
            return noniid_dirichlet_split(dataset, args.num_users, alpha)
        else:
            return noniid_shard_split(dataset, args.num_users)


# FEMNIST

class FEMNISTDataset(Dataset):
    """
    FEMNIST dataset wrapper class.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: (N, 784) numpy array or list, each row is a 28x28 flattened pixel value
            labels: (N,) list of labels
            transform: Optional image transform
        """
        self.images = images  # numpy array (N, 784)
        self.labels = labels  # list or numpy array
        self.targets = list(labels)  # Compatible with sampling module
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Reshape 784-dimensional vector to (28, 28) and convert to FloatTensor
        img = np.array(self.images[idx], dtype=np.float32).reshape(28, 28)
        label = int(self.labels[idx])

        # Convert to (1, 28, 28) tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img, label


def _load_femnist(args):
    """
    Load FEMNIST dataset (LEAF format).
    """
    data_dir = os.path.join('.', 'data', 'leaf_femnist', 'leaf-master', 'data', 'femnist',
                            'data', 'sampled_data')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"FEMNIST data directory does not exist: {data_dir}\n"
            f"Please ensure LEAF FEMNIST data files are placed in the correct directory."
        )

    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    if not json_files:
        raise FileNotFoundError(f"FEMNIST data directory is empty: {data_dir}")

    all_users = []
    all_user_data = {}
    for jf in json_files:
        with open(os.path.join(data_dir, jf), 'r') as f:
            data = json.load(f)
        for user_id in data['users']:
            all_users.append(user_id)
            all_user_data[user_id] = data['user_data'][user_id]

    # Combine all data (train+test) and create original partition by user
    all_images = []
    all_labels = []
    user_train_indices = {}  # Train sample indices for each original user
    user_test_indices = {}   # Test sample indices for each original user
    offset = 0

    for user_id in all_users:
        x = all_user_data[user_id]['x']
        y = all_user_data[user_id]['y']
        n = len(x)
        n_train = max(1, int(0.9 * n))

        perm = np.random.permutation(n)
        x_perm = [x[i] for i in perm]
        y_perm = [y[i] for i in perm]

        user_train_indices[user_id] = list(range(offset, offset + n_train))
        user_test_indices[user_id] = list(range(offset + n_train, offset + n))

        all_images.extend(x_perm)
        all_labels.extend(y_perm)
        offset += n

    all_images = np.array(all_images, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    full_dataset = FEMNISTDataset(all_images, all_labels)

    train_indices = []
    test_indices = []
    for user_id in all_users:
        train_indices.extend(user_train_indices[user_id])
        test_indices.extend(user_test_indices[user_id])

    train_images = all_images[train_indices]
    train_labels = all_labels[train_indices]
    test_images = all_images[test_indices]
    test_labels = all_labels[test_indices]

    train_dataset = FEMNISTDataset(train_images, train_labels)
    test_dataset = FEMNISTDataset(test_images, test_labels)

    if args.iid:
        user_groups = iid_split(train_dataset, args.num_users)
    else:
        # Non-IID mode: Retain the natural user partition of FEMNIST
        num_original_users = len(all_users)
        num_clients = args.num_users

        global_to_train = {gi: ti for ti, gi in enumerate(train_indices)}

        if num_clients <= num_original_users:
            user_groups = {}
            users_per_client = num_original_users // num_clients
            remaining = num_original_users % num_clients

            user_idx = 0
            for cid in range(num_clients):
                n_assign = users_per_client + (1 if cid < remaining else 0)
                client_indices = []
                for j in range(n_assign):
                    if user_idx < num_original_users:
                        uid = all_users[user_idx]
                        for gi in user_train_indices[uid]:
                            if gi in global_to_train:
                                client_indices.append(global_to_train[gi])
                        user_idx += 1
                user_groups[cid] = set(client_indices)
        else:

            alpha = getattr(args, 'alpha', None)
            if alpha is not None:
                user_groups = noniid_dirichlet_split(train_dataset, num_clients, alpha)
            else:
                user_groups = noniid_shard_split(train_dataset, num_clients)

    args.num_classes = 62

    print(f"FEMNIST: Original user count={len(all_users)}, Client count={args.num_users}, "
          f"Training samples={len(train_dataset)}, Testing samples={len(test_dataset)}, Class count=62")

    return train_dataset, test_dataset, user_groups


# Shakespeare
_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ShakespeareDataset(Dataset):
    """
    Character-level Shakespeare dataset.
    """

    def __init__(self, data_tensor, target_tensor):

        self.data = data_tensor
        self.target = target_tensor
        # Pseudo labels: Take the index of the most frequent character in each sample's target sequence
        # For compatibility with noniid_shard_split (shard by .targets)
        self.targets = self.target[:, 0].numpy().tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def _load_shakespeare(args):
    """
    Load Shakespeare dataset (character-level next-char prediction).
    """
    import urllib.request

    data_dir = os.path.join('.', 'data', 'shakespeare')
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, 'input.txt')

    # 1. Download text (if not exists)
    if not os.path.exists(txt_path):
        print(f"Downloading Shakespeare dataset to {txt_path} ...")
        urllib.request.urlretrieve(_SHAKESPEARE_URL, txt_path)
        print("Download complete.")

    with open(txt_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    chars = sorted(set(raw_text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}

    args.vocab_size = vocab_size

    encoded = [char_to_idx[c] for c in raw_text]

    seq_len = args.seq_len
    xs, ys = [], []
    for i in range(0, len(encoded) - seq_len, seq_len):
        xs.append(encoded[i: i + seq_len])
        ys.append(encoded[i + 1: i + seq_len + 1])

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)


    n = len(xs)
    n_train = int(0.9 * n)


    perm = torch.randperm(n)
    xs, ys = xs[perm], ys[perm]

    train_ds = ShakespeareDataset(xs[:n_train], ys[:n_train])
    test_ds = ShakespeareDataset(xs[n_train:], ys[n_train:])


    user_groups = _split_shakespeare(train_ds, args)

    print(f"Shakespeare: vocab_size={vocab_size}, seq_len={seq_len}, "
          f"Training samples={len(train_ds)}, Testing samples={len(test_ds)}")

    return train_ds, test_ds, user_groups


def _split_shakespeare(dataset, args):
    """Client partitioning for Shakespeare dataset"""
    if args.iid:
        return iid_split(dataset, args.num_users)
    else:
        alpha = getattr(args, 'alpha', None)
        if alpha is not None:
            return noniid_dirichlet_split(dataset, args.num_users, alpha)
        else:
            return iid_split(dataset, args.num_users)
