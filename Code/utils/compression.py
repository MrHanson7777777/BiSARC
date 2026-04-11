#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

# Top-K sparsification

def topk_compress(residual, compression_ratio):
    compressed = {}
    for key, param in residual.items():
        if not param.dtype.is_floating_point:
            compressed[key] = param.clone()
            continue
        with torch.no_grad():
            flat = param.view(-1)
            k = max(1, int(len(flat) * compression_ratio))
            if k >= len(flat):
                compressed[key] = param.clone()
                continue
            threshold = torch.kthvalue(torch.abs(flat), len(flat) - k)[0]
            mask = torch.abs(flat) >= threshold
            out = torch.zeros_like(flat)
            out[mask] = flat[mask]
            compressed[key] = out.view(param.shape)
    return compressed


# Bit-level mask packing / unpacking

def pack_sparse(compressed_dict):
    packed = {}
    for key, tensor in compressed_dict.items():
        if not tensor.dtype.is_floating_point:
            continue
        mask = (tensor != 0)
        values = tensor[mask]
        if values.numel() == 0:
            continue
        flat_mask = mask.flatten().cpu().numpy().astype(np.uint8)
        packed_bytes = np.packbits(flat_mask)
        packed[key] = {
            'mask_data': packed_bytes,
            'mask_shape': mask.shape,
            'mask_bits': len(flat_mask),
            'values': values.cpu(),
        }
    return packed


def unpack_sparse(packed_dict, template):
    unpacked = {}
    for key, t in template.items():
        unpacked[key] = torch.zeros_like(t)

    for key, data in packed_dict.items():
        if key not in template:
            continue
        target_device = template[key].device
        bits = np.unpackbits(data['mask_data'])[:data['mask_bits']]
        mask = torch.from_numpy(bits.astype(bool)).reshape(data['mask_shape']).to(target_device)
        values = data['values'].to(target_device)
        unpacked[key][mask] = values

    return unpacked


# Communication volume calculation


def calc_comm_bytes(packed_dict):
    total = 0
    for data in packed_dict.values():
        total += len(data['mask_data'])
        total += data['values'].numel() * data['values'].element_size()
    return total


# QSGD stochastic quantization compression

def stochastic_quantize(tensor, num_bits=8):
    q_levels = 1 << num_bits

    norm = tensor.abs().max()
    if norm == 0:
        return tensor.clone(), 0

    abs_normalized = tensor.abs() / norm
    scaled = abs_normalized * (q_levels - 1)

    floor_val = scaled.floor()
    prob = scaled - floor_val
    mask = torch.rand_like(prob) < prob
    quantized_int = floor_val + mask.float()

    out = tensor.sign() * (quantized_int / (q_levels - 1)) * norm

    total_bits = tensor.numel() * num_bits + 32

    return out, total_bits


def calc_qsgd_bytes(state_dict, num_bits=8):
    total_bits = 0
    for p in state_dict.values():
        if p.dtype.is_floating_point:
            total_bits += p.numel() * num_bits + 32
        else:
            total_bits += p.numel() * p.element_size() * 8
    return total_bits / 8.0
