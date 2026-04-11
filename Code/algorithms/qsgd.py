#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSGD Client —— Stochastic quantization compression baseline
"""

import copy
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.fedavg import FedAvgClient
from utils.compression import stochastic_quantize


class QSGDClient(FedAvgClient):
    def local_train(self, global_weights, global_round):
        """
        Args:
            global_weights: Global model state_dict
            global_round: Current global round

        Returns:
            (quantized_delta, avg_loss, train_acc)
            Note: Returns the quantized increment delta, not the complete state_dict
        """
        model = self._build_local_model(global_weights)
        model.train()

        optimizer = self._make_optimizer(model)
        self._adjust_lr(optimizer, global_round)

        epoch_loss = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if outputs.dim() == 3:
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)),
                                          labels.reshape(-1))
                else:
                    loss = self.criterion(outputs, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        train_acc = self.evaluate_on_training_data(model)

        new_weights = model.state_dict()
        quantized_delta = {}

        num_bits = getattr(self.args, 'qsgd_bits', 8)

        for k in new_weights:
            diff = new_weights[k].cpu() - global_weights[k].cpu()
            if diff.dtype.is_floating_point:
                q_diff, _ = stochastic_quantize(diff, num_bits=num_bits)
                quantized_delta[k] = q_diff
            else:
                quantized_delta[k] = diff

        return quantized_delta, sum(epoch_loss) / len(epoch_loss), train_acc
