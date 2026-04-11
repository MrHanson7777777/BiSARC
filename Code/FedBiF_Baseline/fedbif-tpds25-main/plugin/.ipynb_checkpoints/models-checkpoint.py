import torch
import torch.nn as nn

from model.cnn import CNN_SVHN, CNN_CIFAR, CNN_MNIST
from model.resnet import ResNet18,ResNet101
from model.lstm import CharLSTM
from plugin.bif import bif_replace_modules


def _weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

def _get_conv2d_keys(model):
    keys_weight = []
    for k, val in model.named_modules():
        if isinstance(val, (torch.nn.Conv2d)):
            keys_weight.append(k+".weight")
    return keys_weight

def get_model(model_name, dataset):
    if dataset == "mnist" or dataset == "fmnist":
        if "cnn" in model_name:
            model = CNN_MNIST()
    elif dataset == "femnist":
        # FEMNIST: 62 classes, 28x28 grayscale
        if "cnn" in model_name:
            model = CNN_MNIST(num_classes=62)
        elif "resnet18" in model_name:
            model = ResNet18(num_classes=62, image_size=28, in_channels=1)
        elif "resnet101" in model_name:
            model = ResNet101(num_classes=62, image_size=28, in_channels=1)
    elif dataset == "shakespeare":
        # Shakespeare: next-char prediction, 80 classes
        if "lstm" in model_name:
            model = CharLSTM(num_chars=80, embed_dim=8, hidden_dim=256, num_layers=2)
        else:
            model = CharLSTM(num_chars=80, embed_dim=8, hidden_dim=256, num_layers=2)
    elif dataset == "svhn":
        if "cnn" in model_name:
            model = CNN_SVHN()
    elif dataset == "cifar10":
        if "cnn" in model_name:
            model = CNN_CIFAR(10)
        elif "resnet18" in model_name:
            model = ResNet18(10)
        elif "resnet101" in model_name:
            model = ResNet101(10)
    elif dataset == "cifar100":
        if "cnn" in model_name:
            model = CNN_CIFAR(100)
        elif "resnet18" in model_name:
            model = ResNet18(100)
        elif "resnet101" in model_name:
            model = ResNet101(100)
    elif dataset == "tinyimagenet":
        if "resnet18" in model_name:
            model = ResNet18(200, 64)
        elif "resnet101" in model_name:
            model = ResNet101(200, 64)
    else:
        raise ValueError("wrong dataset.")
    _weight_init(model)
    keys_weight = _get_conv2d_keys(model)
    return model, keys_weight

def get_model_with_config(model_name, dataset, config=None):
    model, keys_weight = get_model(model_name, dataset)
    if config["com_type"] == "fedbif":
        bif_replace_modules(model)
    _weight_init(model)
    return model, keys_weight


if __name__ == "__main__":
    x = torch.arange(1 * 3 * 224 * 224).reshape(1, 3, 224, 224)
    print(f"x shape: {x.shape}")
