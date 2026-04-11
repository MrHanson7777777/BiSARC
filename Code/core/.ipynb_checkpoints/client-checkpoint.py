#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Client base class"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.sampling import DatasetSplit


class BaseClient:
    """
    Base class for all federated learning clients.
    Subclasses must implement local_train() method.
    """

    def __init__(self, args, dataset, idxs, model):
        self.args = args
        self.device = self._get_device(args)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        idxs = list(idxs)
        n = len(idxs)
        self.trainloader = DataLoader(
            DatasetSplit(dataset, idxs[:int(0.8 * n)]),
            batch_size=args.local_bs, shuffle=True)
        self.testloader = DataLoader(
            DatasetSplit(dataset, idxs[int(0.9 * n):]),
            batch_size=max(1, int(0.1 * n) // 10), shuffle=False)


    @staticmethod
    def _get_device(args):
        if args.gpu is not None and torch.cuda.is_available():
            try:
                gpu_id = int(args.gpu)
                torch.randn(1).cuda(gpu_id)
                return torch.device(f'cuda:{gpu_id}')
            except Exception:
                pass
        return torch.device('cpu')

    def _make_optimizer(self, model):
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        else:  # sgd
            return torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)

    def _adjust_lr(self, optimizer, global_round):
        scheduler = getattr(self.args, 'lr_scheduler', 'none')
        if scheduler != 'cosine':
            return
        total = self.args.epochs
        warmup = min(5, total // 10)
        min_lr = self.args.lr * 0.05
        if global_round < warmup:
            lr = self.args.lr * (global_round + 1) / warmup
        else:
            eff = global_round - warmup
            eff_total = total - warmup
            lr = min_lr + (self.args.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * eff / eff_total))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    def inference(self, model):
        # Evaluate model on local test set
        model.to(self.device).eval()
        correct, total, loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss_sum += self.criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / max(1, len(self.testloader))
        return acc, avg_loss

    def evaluate_on_training_data(self, model):
        model.to(self.device).eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0

    def local_train(self, global_weights, global_round):
        """
        Execute local training and return update results.

        Returns:
            (updated results, average loss)
            - FedAvg / FedProx: Returns (state_dict, loss)
            - Residual:         Returns (compressed_update, loss)
        """
        raise NotImplementedError
