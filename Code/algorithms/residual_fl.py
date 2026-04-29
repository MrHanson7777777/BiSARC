#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bidirectional Residual Compression Federated Learning Client
Process:
  1. Receive server downlink broadcast (compressed residual) → Decompress → Update local synchronized model
  2. Start local training from the synchronized model → Obtain updated_weights
  3. Calculate uplink residual = updated_weights - synced_model
  4. Error Feedback (optional): compensated = residual + error_buffer
  5. Top-K Compression → Pack → Upload
  6. Update error_buffer = compensated - compressed (the unsent part)
"""

import copy
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# NOTE: `BaseClient` (in `Code/core/client.py`) defines helper methods like
# `evaluate_on_training_data`, which ResidualClient relies on.
from core.client import BaseClient
from utils.compression import topk_compress, pack_sparse


class ResidualClient(BaseClient):
    """
        synced_model : dict  — Local synchronized model accumulated via compressed residual restoration
        error_buffer : dict  — Uplink error feedback buffer
    """

    def __init__(self, args, dataset, idxs, model):
        super().__init__(args, dataset, idxs, model)

        # Initialize with zeros (global model will be injected gradually during cold start)
        template = model.state_dict()
        self.synced_model = {k: torch.zeros_like(v, device='cpu')
                            for k, v in template.items()}
        
        self.disable_sync = getattr(args, 'disable_sync', 0)
        if self.disable_sync:
            # When Sync is disabled, use local_model as starting point for training
            self.local_model = {k: torch.zeros_like(v, device='cpu')
                                for k, v in template.items()}

        use_ef = getattr(args, 'use_ef', 1)
        if use_ef:
            self.error_buffer = {k: torch.zeros_like(v, device='cpu')
                                for k, v in template.items()}
        else:
            self.error_buffer = None


    def apply_downlink(self, delta):
        """
        Add the increment broadcasted by server (dense dict) to synced_model.
        delta can be the unpacked dense dict or an uncompressed dense dict.
        """
        for k in self.synced_model:
            if k in delta:
                self.synced_model[k] = self.synced_model[k] + delta[k].cpu()
        if self.disable_sync and hasattr(self, 'local_model'):
            for k in self.local_model:
                if k in delta:
                    self.local_model[k] = self.local_model[k] + delta[k].cpu()

    def set_synced_model(self, weights):
        self.synced_model = {k: v.cpu().clone() for k, v in weights.items()}
        # -Sync Mode: Initialize local model as well
        if self.disable_sync and hasattr(self, 'local_model'):
            self.local_model = {k: v.cpu().clone() for k, v in weights.items()}


    def local_train(self, global_weights, global_round):
        """
        Perform local training and return the compressed uplink residual (packed format).

        In residual mode, self.synced_model is used as the starting point for training,
        global_weights parameter is not used.
        In -Sync ablation mode, self.local_model is used as the starting point for training.
        """
        if self.disable_sync and hasattr(self, 'local_model'):
            train_base = self.local_model
        else:
            train_base = self.synced_model

        model = self._build_local_model(train_base)
        model.train()

        optimizer = self._make_optimizer(model)
        self._adjust_lr(optimizer, global_round)

        cr_up = self.args.cr_up  # Use cr_up to control uplink compression
        use_compression = cr_up is not None and 0 < cr_up < 1
        use_ef = getattr(self.args, 'use_ef', 1)  

        epoch_loss = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                
                # NLP sequence tasks: flatten tensors to calculate Loss
                if logits.dim() == 3:
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                          labels.reshape(-1))
                else:
                    loss = self.criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        avg_loss = sum(epoch_loss) / len(epoch_loss)

        # Some environments may run with an older BaseClient that doesn't provide
        # `evaluate_on_training_data` (historical compatibility). Provide a safe fallback.
        train_acc = self.evaluate_on_training_data(model)

        updated = {k: v.cpu() for k, v in model.state_dict().items()}
        residual = _dict_sub(updated, train_base)

        # -Sync Mode: Keep full training result as starting point for next round's local model
        if self.disable_sync and hasattr(self, 'local_model'):
            self.local_model = {k: v.clone() for k, v in updated.items()}

        # Save post-training weight snapshot
        self._last_trained_weights = updated

        if not use_compression:
            return residual, avg_loss, train_acc

        if use_ef and self.error_buffer is not None:
            # With error feedback: compensated = residual + error_buffer
            compensated = _dict_add(residual, self.error_buffer)
        else:
            # Without error feedback: compress raw residual directly
            compensated = residual

        # Top-K compression
        compressed = topk_compress(compensated, cr_up)

        if use_ef and self.error_buffer is not None:
            self.error_buffer = _dict_sub(compensated, compressed)

        packed = pack_sparse(compressed)
        return packed, avg_loss, train_acc


    def evaluate_on_training_data(self, model):
        """Fallback training accuracy evaluation.

        Newer `core.client.BaseClient` already provides this method.
        We keep a local implementation here to avoid runtime breakage when
        the base class version differs.
        """
        model.to(self.device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                # NLP sequence tasks: flatten tensors
                if outputs.dim() == 3:
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    labels = labels.reshape(-1)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0


    def _build_local_model(self, weights):
        from models.cnn import CNNMnist, CNNCifar, CNNFemnist
        from models.resnet import ResNet18Fed

        if self.args.model == 'cnn':
            if self.args.dataset == 'mnist':
                m = CNNMnist(num_classes=self.args.num_classes)
            elif self.args.dataset == 'femnist':
                m = CNNFemnist(num_classes=self.args.num_classes)
            else:
                m = CNNCifar(num_classes=self.args.num_classes)
        elif self.args.model == 'resnet18':
            norm_type = 'groupnorm' if not self.args.iid else 'groupnorm'  # Unified use of GroupNorm
            in_channels = 1 if self.args.dataset in ('mnist', 'femnist') else 3
            m = ResNet18Fed(num_classes=self.args.num_classes, norm_type=norm_type, in_channels=in_channels)
        elif self.args.model == 'lstm':
            from models.rnn import CharLSTM
            m = CharLSTM(vocab_size=self.args.vocab_size,
                         embed_dim=self.args.embed_dim,
                         hidden_dim=self.args.hidden_dim)
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")

        w = {k: v.to(self.device) for k, v in weights.items()}
        m.load_state_dict(w)
        m.to(self.device)
        return m

    def get_train_base_weights(self):
        # Return current training base weights
        if self.disable_sync and hasattr(self, 'local_model'):
            return {k: v.clone() for k, v in self.local_model.items()}
        return {k: v.clone() for k, v in self.synced_model.items()}

    def get_local_weights_after_train(self):
        # Return weights after current round of local_train
        if not hasattr(self, '_last_trained_weights'):
            return None
        return {k: v.clone() for k, v in self._last_trained_weights.items()}


def _dict_sub(d1, d2):
    """d1 - d2"""
    r = {}
    for k in d1:
        t1, t2 = d1[k], d2[k]
        if t1.device != t2.device:
            t2 = t2.to(t1.device)
        r[k] = t1 - t2
    return r


def _dict_add(d1, d2):
    """d1 + d2"""
    r = {}
    for k in d1:
        if k not in d2:
            r[k] = d1[k].clone()
            continue
        t1, t2 = d1[k], d2[k]
        if t1.device != t2.device:
            t2 = t2.to(t1.device)
        r[k] = t1 + t2
    return r