#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Communication overhead comparison experiment.

This script is a *standalone* entry to reproduce the communication overhead
comparison across three algorithms:

- bisarc: bidirectional Top-K residual compression, uplink/downlink CR = 0.1
- fedavg: 32-bit dense uplink + downlink broadcast to all clients
- doublesqueeze: 4-bit stochastic quantization (with error feedback)

It outputs two figures:

Figure 1: Single-round peak bandwidth requirement across communication rounds
          (including t=0 cold start initialization round).

Figure 2: Cumulative communication volume during the main training phase
          (excluding the initialization round, starting from round 1).

Notes
-----
This file intentionally focuses on *communication accounting and plotting*.
It runs a lightweight FL loop similar to `main.py`, but it doesn't aim to be
the canonical training entry.
"""

from __future__ import annotations

import os
import sys
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import csv

import numpy as np
import torch


# Ensure `Code/` is on PYTHONPATH when run from the repository root or `Code/`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)


from utils.options import args_parser
from utils.compression import unpack_sparse, calc_comm_bytes, calc_qsgd_bytes
from data.dataset import get_dataset

from core.server import FedAvgServer
from core.server import ResidualServer

from algorithms.fedavg import FedAvgClient
from algorithms.residual_fl import ResidualClient
from algorithms.doublesqueeze import DoubleSqueezeClient, DoubleSqueezeServer


def build_model(args):
    """Copy of `main.build_model` (kept local to avoid side effects)."""
    from models.cnn import CNNMnist, CNNCifar, CNNFemnist
    from models.resnet import ResNet18Fed
    from models.rnn import CharLSTM

    if args.model == 'cnn':
        if args.dataset == 'mnist':
            model = CNNMnist(num_classes=args.num_classes)
        elif args.dataset == 'femnist':
            model = CNNFemnist(num_classes=args.num_classes)
        else:
            model = CNNCifar(num_classes=args.num_classes)
    elif args.model == 'resnet18':
        norm_type = 'groupnorm' if not args.iid else 'groupnorm'
        in_channels = 1 if args.dataset in ('mnist', 'femnist') else 3
        model = ResNet18Fed(
            num_classes=args.num_classes,
            norm_type=norm_type,
            in_channels=in_channels,
        )
    elif args.model == 'lstm':
        model = CharLSTM(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model


def get_model_byte_size(state_dict: Dict[str, torch.Tensor]) -> int:
    total_bytes = 0
    for t in state_dict.values():
        total_bytes += t.numel() * t.element_size()
    return int(total_bytes)


@dataclass
class CommSeries:
    rounds: List[int]
    peak_bytes: List[float]              # includes t=0
    main_cum_bytes: List[float]          # excludes t=0; series length == rounds length


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _ensure_num_classes(args, train_dataset=None):
    # Match `main.py` behavior.
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'femnist':
        args.num_classes = 62
    elif args.dataset == 'shakespeare':
        args.num_classes = getattr(args, 'vocab_size', 62)
    else:
        args.num_classes = 10


def simulate_comm_overhead(args, alg: str) -> CommSeries:
    """Run a minimal FL loop and record the two required communication series."""

    # Make a per-alg copy of args to avoid cross-run mutation.
    a = copy.deepcopy(args)
    a.alg = alg

    # Hard constraints from the request.
    if alg == 'bisarc':
        a.cr_up = 0.1
        a.cr_down = 0.1
        # `ResidualServer` uses `tx_bits` for values precision when packing.
        # Keep it default 32 unless user overrides.
    elif alg == 'doublesqueeze':
        # DoubleSqueeze uses `doublesqueeze_bits` (not in options.py by default).
        setattr(a, 'doublesqueeze_bits', 4)
        # Ensure error feedback on by default to match typical DoubleSqueeze.
        a.use_ef = 1
    elif alg == 'fedavg':
        # Dense 32-bit is already the default PyTorch parameter dtype.
        pass
    else:
        raise ValueError(f"Unsupported alg for this experiment: {alg}")

    _seed_all(a.seed)
    _ensure_num_classes(a)

    device = torch.device(
        f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu'
    )

    train_dataset, _test_dataset, user_groups = get_dataset(a)
    if a.dataset == 'shakespeare':
        a.num_classes = a.vocab_size

    global_model = build_model(a)

    # Server + clients
    if alg == 'bisarc':
        server = ResidualServer(global_model, a)
        clients = {
            uid: ResidualClient(a, train_dataset, user_groups[uid], global_model)
            for uid in range(a.num_users)
        }
    elif alg == 'fedavg':
        server = FedAvgServer(global_model, a)
        clients = {
            uid: FedAvgClient(a, train_dataset, user_groups[uid], global_model)
            for uid in range(a.num_users)
        }
    else:  # doublesqueeze
        server = DoubleSqueezeServer(global_model, a)
        clients = {
            uid: DoubleSqueezeClient(a, train_dataset, user_groups[uid], global_model)
            for uid in range(a.num_users)
        }

    # --- t=0 cold start: initial model distribution to all clients ---
    # Figure 1 is about **single-round peak bandwidth requirement**.
    # Here we treat "peak" as the *largest one-shot message size* that must be
    # delivered in that round.
    # - For BiSARC (streaming init): peak = max(streaming-step message)
    # - For fedavg/doublesqueeze: peak = full initialization broadcast size
    total_comm_bytes = 0.0
    main_cum_comm_bytes = 0.0
    peak_series: List[float] = []
    main_cum_series: List[float] = []

    gw = server.get_global_weights()

    if alg == 'bisarc':
        # We follow the same convention as `main.py`:
        # residual may have streaming init when downlink compression is enabled.
        use_compression_down = a.cr_down is not None and 0 < a.cr_down < 1
        cold_total = 0.0
        cold_peak = 0.0

        init_rounds = server.init_stream_rounds()
        if init_rounds > 0 and use_compression_down:
            for _ in range(init_rounds):
                packed_delta = server.init_stream_step()
                step_bytes_per_client = calc_comm_bytes(packed_delta)
                step_total = step_bytes_per_client * a.num_users
                cold_total += step_total
                cold_peak = max(cold_peak, step_total)
                # Apply to all clients to keep protocol consistent.
                template = {k: torch.zeros_like(v, device='cpu') for k, v in gw.items()}
                dense_delta = unpack_sparse(packed_delta, template)
                for uid in range(a.num_users):
                    clients[uid].apply_downlink(dense_delta)
        else:
            cold_total = get_model_byte_size(gw) * a.num_users
            cold_peak = cold_total
            for uid in range(a.num_users):
                clients[uid].set_synced_model(gw)

        total_comm_bytes += cold_total
        peak_series.append(cold_peak)
        main_cum_series.append(0.0)

    elif alg == 'doublesqueeze':
        # Cold start peak requirement: in practice, server still needs to deliver
        # the initialized model to all clients once.
        # To keep Figure-1 t=0 comparable across baselines, we treat t=0 as a
        # full-model broadcast (same as FedAvg). Quantization affects *training
        # phase* traffic.
        cold_per_client = get_model_byte_size(gw)
        cold_total = float(cold_per_client) * a.num_users
        total_comm_bytes += cold_total
        # One-shot cold start broadcast, peak equals the full broadcast volume.
        peak_series.append(cold_total)
        main_cum_series.append(0.0)
    else:  # fedavg
        cold_per_client = get_model_byte_size(gw)
        cold_total = float(cold_per_client) * a.num_users
        total_comm_bytes += cold_total
        # One-shot cold start broadcast, peak equals the full broadcast volume.
        peak_series.append(cold_total)
        main_cum_series.append(0.0)

    # --- main training rounds ---
    # IMPORTANT (per user's round definition):
    # round1/round2/... all include:
    #   (1) clients train + upload (uplink)
    #   (2) server aggregates
    #   (3) server broadcasts the *current round's* aggregated payload (downlink)
    # Therefore, for `bisarc` and `doublesqueeze`, the downlink payload is produced
    # by this round's `server.aggregate(...)`, not a cached payload from the
    # previous round.
    use_compression_up = (alg == 'bisarc') and (a.cr_up is not None and 0 < a.cr_up < 1)
    use_compression_down = (alg == 'bisarc') and (a.cr_down is not None and 0 < a.cr_down < 1)

    for epoch in range(a.epochs):
        m = max(int(a.frac * a.num_users), 1)
        selected = random.sample(range(a.num_users), m)
        global_weights = server.get_global_weights()

        # Uplink from selected clients.
        local_results = []
        client_data_sizes = []
        uplink_bytes_this_round = 0.0

        for uid in selected:
            result, _loss, _train_acc = clients[uid].local_train(global_weights, epoch)
            local_results.append(result)
            client_data_sizes.append(len(user_groups[uid]))

            if alg == 'bisarc':
                if use_compression_up:
                    uplink_bytes_this_round += calc_comm_bytes(result)
                else:
                    uplink_bytes_this_round += get_model_byte_size(result)
            elif alg == 'doublesqueeze':
                bits = getattr(a, 'doublesqueeze_bits', 4)
                uplink_bytes_this_round += calc_qsgd_bytes(result, bits)
            else:
                uplink_bytes_this_round += get_model_byte_size(result)

        # Server side aggregation.
        # The returned value (if any) is the payload to be broadcast in *this* round.
        downlink_payload = None
        if alg == 'bisarc':
            downlink_payload = server.aggregate(local_results, client_data_sizes)
        elif alg == 'doublesqueeze':
            downlink_payload = server.aggregate(local_results, client_data_sizes)
        else:
            server.aggregate(local_results, client_data_sizes)
            downlink_payload = server.get_global_weights()

        # Downlink billing + applying downlink to all clients (this round's broadcast).
        downlink_bytes_this_round = 0.0
        if alg == 'bisarc':
            # ResidualServer.aggregate returns packed sparse when cr_down in (0,1).
            msg_size = (
                calc_comm_bytes(downlink_payload)
                if use_compression_down
                else get_model_byte_size(downlink_payload)
            )
            downlink_bytes_this_round = float(msg_size) * a.num_users

            template = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
            dense_down = (
                unpack_sparse(downlink_payload, template)
                if use_compression_down
                else downlink_payload
            )
            for uid in range(a.num_users):
                clients[uid].apply_downlink(dense_down)
        elif alg == 'doublesqueeze':
            bits = getattr(a, 'doublesqueeze_bits', 4)
            downlink_bytes_this_round = calc_qsgd_bytes(downlink_payload, bits) * a.num_users
        else:  # fedavg
            downlink_bytes_this_round = get_model_byte_size(downlink_payload) * a.num_users

        total_comm_bytes += (uplink_bytes_this_round + downlink_bytes_this_round)

        main_cum_comm_bytes += (uplink_bytes_this_round + downlink_bytes_this_round)

        # Record this round (t = epoch + 1)
        # Single-round peak bandwidth requirement (Figure 1) is the larger
        # one-shot transfer between downlink broadcast and uplink (all selected).
        peak_bytes = float(max(uplink_bytes_this_round, downlink_bytes_this_round))
        peak_series.append(peak_bytes)
        main_cum_series.append(main_cum_comm_bytes)

    # Use real round indices: t=0 is cold start, then t=1..E are training rounds.
    rounds = list(range(0, a.epochs + 1))
    if not (len(peak_series) == len(rounds) == len(main_cum_series)):
        raise RuntimeError(
            f"CommSeries length mismatch for {alg}: rounds={len(rounds)}, "
            f"peak={len(peak_series)}, main_cum={len(main_cum_series)}"
        )
    return CommSeries(rounds=rounds, peak_bytes=peak_series, main_cum_bytes=main_cum_series)


def _plot_results(out_dir: str, results: Dict[str, CommSeries]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Colors are fixed for comparability.
    style = {
        'bisarc': dict(color='#d62728', linewidth=2.2),
        'fedavg': dict(color='#1f77b4', linewidth=2.0),
        'doublesqueeze': dict(color='#2ca02c', linewidth=2.0),
    }

    # Figure 1: peak bytes per round (includes t=0).
    fig1 = plt.figure(figsize=(7.5, 4.2))
    ax1 = fig1.add_subplot(1, 1, 1)
    for alg, series in results.items():
        ax1.plot(
            series.rounds,
            np.array(series.peak_bytes) / (1024.0 * 1024.0),
            label=alg,
            **style.get(alg, {}),
        )
    ax1.set_xlabel('Communication round (t)')
    ax1.set_ylabel('Peak bandwidth per round (MB)')
    ax1.set_title('Figure 1: Single-round peak bandwidth requirement')
    ax1.axvline(0, color='k', linewidth=1.0, alpha=0.25)
    ax1.text(0, ax1.get_ylim()[1] * 0.98, 't=0 (init)', ha='left', va='top', fontsize=9, alpha=0.8)
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # Force integer ticks for rounds to avoid 0.00 ~ 2.00 style display.
    max_t = max((max(s.rounds) for s in results.values()), default=0)
    ax1.set_xticks(list(range(0, int(max_t) + 1, max(1, int(max_t // 10) or 1))))
    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, 'figure1_peak_bandwidth.png')
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # Figure 2: main phase cumulative bytes (exclude t=0).
    fig2 = plt.figure(figsize=(7.5, 4.2))
    ax2 = fig2.add_subplot(1, 1, 1)
    for alg, series in results.items():
        rounds_main = series.rounds[1:]
        cum_main = np.array(series.main_cum_bytes[1:]) / (1024.0 * 1024.0)
        ax2.plot(rounds_main, cum_main, label=alg, **style.get(alg, {}))
    ax2.set_xlabel('Communication round (t), main phase (t>=1)')
    ax2.set_ylabel('Cumulative communication volume (MB)')
    ax2.set_title('Figure 2: Cumulative communication volume (main phase)')
    max_t2 = max((max(s.rounds) for s in results.values()), default=0)
    ax2.set_xticks(list(range(1, int(max_t2) + 1, max(1, int(max_t2 // 10) or 1))))
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, 'figure2_cumulative_main_phase.png')
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    return fig1_path, fig2_path


def _save_csv(out_dir: str, results: Dict[str, CommSeries]) -> str:
    """Save per-round communication accounting to a single CSV.

    Columns:
      - round
      - {alg}_peak_mb
      - {alg}_main_cum_mb  (0 at t=0)
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'comm_overhead.csv')

    # Assume all series use the same rounds length (t=0..E)
    rounds = next(iter(results.values())).rounds if results else []
    algs = list(results.keys())

    fieldnames = ['round']
    for alg in algs:
        fieldnames.append(f'{alg}_peak_mb')
        fieldnames.append(f'{alg}_main_cum_mb')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, t in enumerate(rounds):
            row = {'round': t}
            for alg in algs:
                s = results[alg]
                row[f'{alg}_peak_mb'] = float(s.peak_bytes[i]) / (1024.0 * 1024.0)
                row[f'{alg}_main_cum_mb'] = float(s.main_cum_bytes[i]) / (1024.0 * 1024.0)
            w.writerow(row)

    return csv_path


def main():
    args = args_parser()
    # We don't add new CLI flags to avoid touching the global parser for now.
    # Lightweight local verification: set env `BISARC_SMOKE=1` to shorten runtime.
    # Output directory is fixed under `Code/outputs/comm_overhead`.
    # Smoke mode is opt-in via env, but we should NOT override explicit CLI choices.
    # Heuristic: only shrink when user keeps default values.
    if os.environ.get('BISARC_SMOKE', '').strip() in ('1', 'true', 'True'):
        # Default epochs in options.py is 50; only override when still default.
        if getattr(args, 'epochs', 50) == 50:
            args.epochs = 2
        # Default num_users is 100; only override when still default.
        if getattr(args, 'num_users', 100) == 100:
            args.num_users = 5
        # Default frac is 0.1; only override when still default.
        if abs(getattr(args, 'frac', 0.1) - 0.1) < 1e-12:
            args.frac = 1.0
        # Default local_ep is 5; only override when still default.
        if getattr(args, 'local_ep', 5) == 5:
            args.local_ep = 1

    out_dir = os.path.join(_CODE_DIR, 'outputs', 'comm_overhead')

    # Run three algorithms as requested.
    algs = ['bisarc', 'fedavg', 'doublesqueeze']
    results: Dict[str, CommSeries] = {}
    for alg in algs:
        results[alg] = simulate_comm_overhead(args, alg)

    fig1_path, fig2_path = _plot_results(out_dir, results)
    csv_path = _save_csv(out_dir, results)
    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig2_path}")
    print(f"Saved: {csv_path}")
    print(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()
