#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FedAvg Aggregation & Residual Compression Aggregation"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.compression import unpack_sparse, topk_compress, pack_sparse


class FedAvgServer:
    # Standard FedAvg / FedProx Server: Receives full weights, performs weighted average

    def __init__(self, global_model, args):
        self.args = args
        self.global_model = global_model
        self.device = self._get_device(args)
        self.global_model.to(self.device)

    @staticmethod
    def _get_device(args):
        if args.gpu is not None and torch.cuda.is_available():
            try:
                return torch.device(f'cuda:{int(args.gpu)}')
            except Exception:
                pass
        return torch.device('cpu')

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, local_weights_list, client_data_sizes):
        total_samples = sum(client_data_sizes)
        weights = [s / total_samples for s in client_data_sizes]

        avg = {}
        for key in local_weights_list[0]:
            avg[key] = torch.zeros_like(local_weights_list[0][key], dtype=torch.float32)
            for i, local_w in enumerate(local_weights_list):
                avg[key] += weights[i] * local_w[key].float()
            # Cast back to original dtype (e.g. int64 for buffers like num_batches_tracked)
            avg[key] = avg[key].to(local_weights_list[0][key].dtype)

        self.global_model.load_state_dict(avg)

    def test(self, test_dataset):
        """Evaluate on the global test set"""
        self.global_model.to(self.device).eval()
        loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        criterion = nn.CrossEntropyLoss().to(self.device)
        correct, total, loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                
                # NLP sequence tasks: flatten tensors
                if outputs.dim() == 3:
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    labels = labels.reshape(-1)

                loss_sum += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total, loss_sum / max(1, len(loader))



class ResidualServer:
    """
    Bidirectional Residual Compression Server.
    Maintains: Global model, downlink error feedback state.
    """

    def __init__(self, global_model, args):
        self.args = args
        self.device = FedAvgServer._get_device(args)
        self.global_model = global_model.to(self.device)

        weights = self.global_model.state_dict()

        # Server-side downlink error feedback buffer
        self.downlink_error = {k: torch.zeros_like(v, device='cpu')
                               for k, v in weights.items()}

        # Synchronized model from the server's perspective (used for downlink streaming)
        self.synced_model = {k: torch.zeros_like(v, device='cpu')
                             for k, v in weights.items()}

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def init_stream_rounds(self):
        cr_down = self.args.cr_down
        if cr_down is None or cr_down <= 0 or cr_down >= 1:
            return 0
        return int(1.0 / cr_down) + 1

    def init_stream_step(self):
        cr_down = self.args.cr_down
        if cr_down is None or cr_down <= 0 or cr_down >= 1:
            return None
        
        target = self.global_model.state_dict()
        residual = _model_sub(target, self.synced_model)
        compressed = topk_compress(residual, cr_down)
        self.synced_model = _model_add(self.synced_model, compressed)
        return pack_sparse(compressed)

    def aggregate(self, packed_residuals, client_data_sizes=None, template=None):
        if template is None:
            template = {k: v.cpu() for k, v in self.global_model.state_dict().items()}

        cr_up = self.args.cr_up
        cr_down = self.args.cr_down
        use_compression = cr_up is not None and 0 < cr_up < 1
        use_compression_down = cr_down is not None and 0 < cr_down < 1
        use_ef = getattr(self.args, 'use_ef', 1)

        dense_list = []
        for pr in packed_residuals:
            if use_compression:
                dense_list.append(unpack_sparse(pr, template))
            else:
                dense_list.append(pr)

        if client_data_sizes is not None:
            total_samples = sum(client_data_sizes)
            weights = [size / total_samples for size in client_data_sizes]
        else:
            weights = [1.0 / len(dense_list)] * len(dense_list)

        avg = {}
        for key in dense_list[0]:
            avg[key] = torch.zeros_like(dense_list[0][key])
            for i, dense_res in enumerate(dense_list):
                avg[key] += weights[i] * dense_res[key]

        cur = self.global_model.state_dict()
        updated = _model_add(cur, avg)
        self.global_model.load_state_dict(updated)

        if use_compression_down:
            if use_ef:
                compensated = _model_add(avg, self.downlink_error)
            else:
                compensated = avg
            
            compressed_down = topk_compress(compensated, cr_down)
            
            if use_ef:
                self.downlink_error = _model_sub(compensated, compressed_down)
            else:
                for k in self.downlink_error:
                    self.downlink_error[k].zero_()
            
            return pack_sparse(compressed_down)
        else:
            return avg

    def test(self, test_dataset):
        self.global_model.to(self.device).eval()
        loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        criterion = nn.CrossEntropyLoss().to(self.device)
        correct, total, loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                
                # NLP sequence tasks: flatten tensors
                if outputs.dim() == 3:
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    labels = labels.reshape(-1)

                loss_sum += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total, loss_sum / max(1, len(loader))



def _model_sub(d1, d2):
    """d1 - d2 (layer-wise)"""
    result = {}
    for k in d1:
        t1, t2 = d1[k], d2[k]
        if t1.device != t2.device:
            t2 = t2.to(t1.device)
        result[k] = t1 - t2
    return result


def _model_add(d1, d2):
    """d1 + d2 (layer-wise), missing keys in d2 retain original values from d1"""
    result = {}
    for k in d1:
        if k not in d2:
            result[k] = d1[k].clone()
            continue
        t1, t2 = d1[k], d2[k]
        if t1.device != t2.device:
            t2 = t2.to(t1.device)
        result[k] = t1 + t2
    return result