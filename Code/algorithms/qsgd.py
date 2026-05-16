#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSGD Client & Server —— Unidirectional stochastic quantization compression baseline

Reference:
    Alistarh et al., "QSGD: Communication-Efficient SGD via Gradient
    Quantization and Encoding", NIPS 2017.

Key properties of QSGD:
    - Uplink-only compression: clients quantize local updates before sending
      to the server; the server aggregates and broadcasts in full precision.
    - Stochastic quantization is unbiased: E[Q_s(v)] = v.
    - No error feedback (error feedback is a DoubleSqueeze addition).
    - Elias coding is omitted; we use a simple fixed-width representation
      whose cost is accounted for via calc_qsgd_bytes().
"""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.fedavg import FedAvgClient
from core.server import FedAvgServer
from utils.compression import stochastic_quantize


class QSGDClient(FedAvgClient):
    """
    QSGD client: local training followed by stochastic quantization of the
    model update (delta = w_new - w_global).  No error feedback.
    """

    def local_train(self, global_weights, global_round):
        """
        Returns:
            (quantized_delta, avg_loss, train_acc)
            quantized_delta is the stochastically quantized (w_new - w_global).
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

            if diff.is_floating_point():
                q_diff, _ = stochastic_quantize(diff, num_bits=num_bits)
                quantized_delta[k] = q_diff
            else:
                quantized_delta[k] = diff

        return quantized_delta, sum(epoch_loss) / len(epoch_loss), train_acc


class QSGDServer(FedAvgServer):
    """
    QSGD server: weighted-average aggregation of quantized client deltas,
    followed by a full-precision global model update.
    No server-side quantization or error feedback (uplink-only compression).
    """

    def aggregate(self, local_weights_list, client_data_sizes):
        """
        Aggregate quantized deltas from clients and update the global model.

        Args:
            local_weights_list: list of quantized_delta dicts from clients
            client_data_sizes:  list of per-client dataset sizes

        Returns:
            aggregated_delta (full-precision) for downlink byte accounting.
        """
        aggregated_delta = {
            k: torch.zeros_like(v).cpu()
            for k, v in self.global_model.state_dict().items()
        }
        total_samples = sum(client_data_sizes)

        for i, local_delta in enumerate(local_weights_list):
            weight = client_data_sizes[i] / total_samples
            for k in aggregated_delta:
                src = local_delta[k].cpu()
                # Integer buffers (e.g. BatchNorm.num_batches_tracked) cannot
                # be accumulated via weighted float arithmetic into an int tensor.
                if not aggregated_delta[k].is_floating_point():
                    tmp = aggregated_delta[k].float() + src.float() * weight
                    aggregated_delta[k] = tmp.round().to(aggregated_delta[k].dtype)
                else:
                    aggregated_delta[k] += src * weight

        global_weights_cpu = {
            k: v.cpu() for k, v in self.global_model.state_dict().items()
        }
        for k in aggregated_delta:
            global_weights_cpu[k] += aggregated_delta[k]

        self.global_model.load_state_dict(global_weights_cpu)

        return aggregated_delta
