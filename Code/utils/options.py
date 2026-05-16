#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def args_parser():
    p = argparse.ArgumentParser()

    # Algorithm selection
    p.add_argument('--alg', type=str, default='fedavg',
                   choices=['fedavg', 'fedprox', 'bisarc', 'qsgd', 'doublesqueeze'],
                   help='Algorithm: fedavg | fedprox | bisarc (BiSARC) | qsgd | doublesqueeze (E=1 bidirectional quantized EF baseline)')

    # Federated learning parameters
    p.add_argument('--epochs', type=int, default=50, help='Global training epochs')
    p.add_argument('--num_users', type=int, default=100, help='Total number of clients')
    p.add_argument('--frac', type=float, default=0.1, help='Fraction of clients selected per round')
    p.add_argument('--local_ep', type=int, default=5, help='Local training epochs')
    p.add_argument('--local_bs', type=int, default=32, help='Local batch size')

    # Optimizer parameters
    p.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    p.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    p.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    p.add_argument('--optimizer', type=str, default='sgd',
                   choices=['sgd', 'adam', 'adamw'], help='Optimizer')
    p.add_argument('--lr_scheduler', type=str, default='none',
                   choices=['none', 'cosine'], help='Learning rate scheduler')

    # Model and dataset
    p.add_argument('--model', type=str, default='cnn',
                   choices=['cnn', 'resnet18', 'lstm'], help='Model')
    p.add_argument('--dataset', type=str, default='mnist',
                   choices=['mnist', 'cifar', 'cifar100', 'femnist', 'shakespeare'], help='Dataset')
    p.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    p.add_argument('--iid', type=int, default=1, help='1=IID, 0=Non-IID')
    p.add_argument('--alpha', type=float, default=None, help='Dirichlet parameter')

    # NLP / Shakespeare parameters
    p.add_argument('--seq_len', type=int, default=80, help='Text sequence length')
    p.add_argument('--hidden_dim', type=int, default=256, help='LSTM hidden dimension')
    p.add_argument('--embed_dim', type=int, default=8, help='Word embedding dimension')
    p.add_argument('--vocab_size', type=int, default=90, help='Dictionary size')

    # Residual compression parameters
    p.add_argument('--cr_up', type=float, default=None, help='Uplink compression ratio')
    p.add_argument('--cr_down', type=float, default=None, help='Downlink compression ratio')
    p.add_argument('--use_ef', type=int, default=1, help='Enable error feedback (1/0)')
    p.add_argument('--tx_bits', type=int, default=32, choices=[16, 32], help='Top-K values precision (16/32)')
    
    # Ablation study parameters
    p.add_argument('--disable_sync', type=int, default=0, help='Disable synchronization replica (1/0)')

    # FedProx parameters
    p.add_argument('--mu', type=float, default=0, help='Proximal term coefficient mu')

    # QSGD parameters
    p.add_argument('--qsgd_bits', type=int, default=8, help='QSGD quantization bits')

    # DoubleSqueeze parameters (bidirectional quantized EF, E=1 single-step baseline)
    p.add_argument('--doublesqueeze_bits', type=int, default=8,
                   help='DoubleSqueeze quantization bits for both uplink and downlink')

    # Others
    p.add_argument('--gpu', type=int, default=None, help='GPU ID')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--stopping_rounds', type=int, default=100, help='Early stopping patience')
    p.add_argument('--verbose', type=int, default=1, help='Print verbose logs')
    p.add_argument('--use_ema', type=int, default=0, help='Use EMA smoothing')
    p.add_argument('--ema_alpha', type=float, default=0.4, help='EMA smoothing coefficient')

    args = p.parse_args()
    return args