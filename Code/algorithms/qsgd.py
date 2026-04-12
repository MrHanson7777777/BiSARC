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
from core.server import FedAvgServer
from utils.compression import stochastic_quantize


class QSGDClient(FedAvgClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_feedback = None
        # Control whether to turn on error feedback (1 means on, which equals to DoubleSqueeze; 0 means off, which equals to pure QSGD)
        self.use_ef = getattr(self.args, 'use_ef', 1)

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

        if self.error_feedback is None:
            self.error_feedback = {k: torch.zeros_like(v).cpu() for k, v in global_weights.items()}

        for k in new_weights:
            diff = new_weights[k].cpu() - global_weights[k].cpu()
            
            if diff.is_floating_point():
                # Add error feedback if use_ef is True
                diff_with_error = diff + self.error_feedback[k].cpu() if self.use_ef else diff
                
                # Quantize
                q_diff, _ = stochastic_quantize(diff_with_error, num_bits=num_bits)
                quantized_delta[k] = q_diff
                
                # Update error feedback (DoubleSqueeze: error = actual - quantized)
                if self.use_ef:
                    self.error_feedback[k] = diff_with_error - q_diff
            else:
                quantized_delta[k] = diff

        return quantized_delta, sum(epoch_loss) / len(epoch_loss), train_acc

class QSGDServer(FedAvgServer):
    def __init__(self, global_model, args):
        super().__init__(global_model, args)
        weights = self.global_model.state_dict()
        self.error_feedback = {k: torch.zeros_like(v).cpu() for k, v in weights.items()}
        self.use_ef = getattr(self.args, 'use_ef', 1)

    def aggregate(self, local_weights_list, client_data_sizes):
        """
        Aggregate local updates and update global weights.
        local_weights_list: list of quantized_delta dictionaries
        """
        aggregated_delta = {k: torch.zeros_like(v).cpu() for k, v in self.global_model.state_dict().items()}
        total_samples = sum(client_data_sizes)
        
        for i, local_delta in enumerate(local_weights_list):
            weight = client_data_sizes[i] / total_samples
            for k in aggregated_delta.keys():
                aggregated_delta[k] += local_delta[k].cpu() * weight

        num_bits = getattr(self.args, 'qsgd_bits', 8)
        
        quantized_global_delta = {}
        global_weights_cpu = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        
        for k in aggregated_delta.keys():
            diff = aggregated_delta[k]
            
            if diff.is_floating_point():
                # Add server-side error feedback if use_ef is True
                diff_with_error = diff + self.error_feedback[k].cpu() if self.use_ef else diff
                
                # Quantize the aggregated delta
                q_diff, _ = stochastic_quantize(diff_with_error, num_bits=num_bits)
                quantized_global_delta[k] = q_diff
                
                # Update server-side error feedback
                if self.use_ef:
                    self.error_feedback[k] = diff_with_error - q_diff
                
                # Update global weights
                global_weights_cpu[k] += q_diff
            else:
                quantized_global_delta[k] = diff
                global_weights_cpu[k] += diff
                
        self.global_model.load_state_dict(global_weights_cpu)
        
        # Return quantized delta for accurate downlink transmission calculation
        return quantized_global_delta
