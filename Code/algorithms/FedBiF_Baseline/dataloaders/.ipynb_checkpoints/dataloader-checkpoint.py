import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
from dataloaders.tinyimagenet import TinyImageNet
from dataloaders.femnist import FemnistDataset, get_femnist_train_data
from dataloaders.shakespeare import ShakespeareDataset, get_shakespeare_train_data


if __name__ == "__main__":
    workspace = "./"
else:
    workspace = "../"

dataset_dict = {
    "mnist": MNIST,
    "fmnist": FashionMNIST,
    "svhn": SVHN,
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "tinyimagenet": TinyImageNet,
    }

transform_dict = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "fmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ]),
    "svhn": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4496,), (0.1995,))
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ]),
    "tinyimagenet": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # FEMNIST: images are already [0,1] tensors from FemnistDataset, only normalize
    "femnist": transforms.Compose([
        transforms.Normalize((0.9637,), (0.1592,))
    ]),
    }

def get_data(dataset):
    if dataset == "femnist":
        # FEMNIST uses LEAF format, load from JSON files
        cache = get_femnist_train_data(seed=1234)
        train = FemnistDataset(cache['train_data'], transform=transform_dict["femnist"])
        return train
    elif dataset == "shakespeare":
        # Shakespeare: next-char prediction, no image transform needed
        cache = get_shakespeare_train_data(seed=1234)
        train = ShakespeareDataset(cache['train_data'])
        return train
    elif "svhn" in dataset:
        train = dataset_dict[dataset](
            root=os.path.join(workspace, "dataset"),
            split="train",
            transform=transform_dict[dataset],
            download=True,
        )
    else:
        train = dataset_dict[dataset](
            root=os.path.join(workspace, "dataset"),
            train=True,
            transform=transform_dict[dataset],
            download=True,
        )
    return train

def get_client_train_dataloader(train, dataset, part_strategy, num_clients, id=0, batch_size=64, val_ratio=0.0, seed=1234):
    # train = get_data(dataset)

    npy_name = f"{dataset}-{part_strategy}-{str(num_clients)}-{str(seed)}.npy"
    train_id_map = np.load(workspace + "/dataloaders/_npy_/" + npy_name, allow_pickle=True)
    train_id_map = train_id_map.item()

    client_data = torch.utils.data.Subset(train, train_id_map[id])
    n_valset = int(len(client_data) * val_ratio)
    valset = torch.utils.data.Subset(client_data, range(0, n_valset))
    trainset = torch.utils.data.Subset(client_data, range(n_valset, len(client_data)))
    valLoader = DataLoader(valset, batch_size=batch_size)
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainLoader, valLoader

def get_server_test_dataloader(dataset, batch_size):
    if dataset == "femnist":
        cache = get_femnist_train_data(seed=1234)
        test = FemnistDataset(cache['test_data'], transform=transform_dict["femnist"])
        test_Loader = DataLoader(test, batch_size=batch_size)
        return test_Loader
    elif dataset == "shakespeare":
        cache = get_shakespeare_train_data(seed=1234)
        test = ShakespeareDataset(cache['test_data'])
        test_Loader = DataLoader(test, batch_size=batch_size)
        return test_Loader
    elif "svhn" in dataset:
        test = dataset_dict[dataset](
            root=os.path.join(workspace, "dataset"),
            split="test",
            transform=transform_dict[dataset],
            download=True,
        )
    else:
        test = dataset_dict[dataset](
            root=os.path.join(workspace, "dataset"),
            train=False,
            transform=transform_dict[dataset],
            download=True,
        )
    test_Loader = DataLoader(test, batch_size=batch_size)
    return test_Loader


if __name__ == "__main__":
    dataset_name = "svhn"
    trainset = get_data(dataset_name)
    print(f"trainset: {len(trainset)=}")
    trainloader, valloader = get_client_train_dataloader(
        trainset,
        dataset=dataset_name,
        part_strategy="iid",
        num_clients=100,
        id=0,
        batch_size=64,
        val_ratio=0.0,
        seed=1234,
    )
    test_loader = get_server_test_dataloader(dataset=dataset_name, batch_size=64)
    print(len(trainloader))
    print(len(valloader))
    print(len(test_loader))
