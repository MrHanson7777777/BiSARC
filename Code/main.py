#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MyFederatedFL -- Unified Entry
"""

import os
import sys
import copy
import math
import time
import random
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.options import args_parser
from utils.metrics import MetricsLogger
from utils.compression import unpack_sparse, calc_comm_bytes, calc_qsgd_bytes
from data.dataset import get_dataset
from models.cnn import CNNMnist, CNNCifar, CNNFemnist
from models.resnet import ResNet18Fed
from core.server import FedAvgServer, ResidualServer
from algorithms.fedavg import FedAvgClient
from algorithms.residual_fl import ResidualClient
from algorithms.qsgd import QSGDClient, QSGDServer


# Model construction

def build_model(args):
    """Build global model according to args"""
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            model = CNNMnist(num_classes=args.num_classes)
        elif args.dataset == 'femnist':
            model = CNNFemnist(num_classes=args.num_classes)
        else:
            model = CNNCifar(num_classes=args.num_classes)
    elif args.model == 'resnet18':
        norm_type = 'groupnorm' if not args.iid else 'groupnorm'  # Unified use of GroupNorm
        model = ResNet18Fed(num_classes=args.num_classes, norm_type=norm_type)
    elif args.model == 'lstm':
        from models.rnn import CharLSTM
        model = CharLSTM(vocab_size=args.vocab_size,
                         embed_dim=args.embed_dim,
                         hidden_dim=args.hidden_dim)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def get_model_byte_size(state_dict):
    """Calculate the total transmission bytes of state_dict (or dense dict of same structure)"""
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


def _aggregate_qsgd_deltas(server, delta_list, client_data_sizes):
    """
    Aggregate the quantized intermediate delta returned by QSGD clients.

    QSGD client returns quantized delta = W_new - W_old (not full weights),
    so we need to perform weighted average on delta first, and then apply it to the global model.

    Args:
        server: FedAvgServer instance
        delta_list: List of quantized deltas from clients [{key: Tensor}, ...]
        client_data_sizes: List of data sizes from all clients
    """
    total_samples = sum(client_data_sizes)
    weights = [s / total_samples for s in client_data_sizes]

    avg_delta = {}
    for key in delta_list[0]:
        avg_delta[key] = torch.zeros_like(delta_list[0][key])
        for i, delta in enumerate(delta_list):
            avg_delta[key] += weights[i] * delta[key].to(avg_delta[key].device)

    cur = server.global_model.state_dict()
    updated = {}
    for key in cur:
        if key in avg_delta:
            d = avg_delta[key]
            if d.device != cur[key].device:
                d = d.to(cur[key].device)
            updated[key] = cur[key] + d
        else:
            updated[key] = cur[key]
    server.global_model.load_state_dict(updated)


# Main training process

def main():
    args = args_parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'femnist':
        args.num_classes = 62
    elif args.dataset == 'shakespeare':
        args.num_classes = getattr(args, 'vocab_size', 62) # This will be properly set in get_dataset later
    else:
        args.num_classes = 10

    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset, user_groups = get_dataset(args)
    if args.dataset == 'shakespeare':
        args.num_classes = args.vocab_size
    print(f"Load dataset: {args.dataset} | Train set: {len(train_dataset)}, Test set: {len(test_dataset)}, Clients: {len(user_groups)}")

    global_model = build_model(args)
    print(f"Model: {args.model}, Params: {sum(p.numel() for p in global_model.parameters()):,}")

    is_residual = (args.alg == 'ours')
    is_qsgd = (args.alg == 'qsgd')
    cr_up = args.cr_up
    cr_down = args.cr_down
    use_compression_up = cr_up is not None and 0 < cr_up < 1
    use_compression_down = cr_down is not None and 0 < cr_down < 1

    if is_residual:
        server = ResidualServer(global_model, args)
    elif is_qsgd:
        server = QSGDServer(global_model, args)
    else:
        server = FedAvgServer(global_model, args)

    clients = {}
    for uid in range(args.num_users):
        if args.alg == 'fedavg':
            clients[uid] = FedAvgClient(args, train_dataset, user_groups[uid], global_model)
        elif args.alg == 'fedprox':
            clients[uid] = FedAvgClient(args, train_dataset, user_groups[uid], global_model)
        elif args.alg == 'ours':
            clients[uid] = ResidualClient(args, train_dataset, user_groups[uid], global_model)
        elif args.alg == 'qsgd':
            clients[uid] = QSGDClient(args, train_dataset, user_groups[uid], global_model)

    logger = MetricsLogger()
    total_comm_bytes = 0  # Cumulative communication bytes (calculated from cold start)
    main_cum_comm_bytes = 0  # Cumulative communication bytes for main training phase (excluding cold start, starting from round 1)
    peak_bytes_list = []  # Save peak bandwidth requirement per round (for debugging/verification)

    ema_acc = None
    best_ema_acc = -1.0
    patience_counter = 0
    patience = args.stopping_rounds
    best_weights = None
    
    use_ema = bool(args.use_ema)
    ema_alpha = args.ema_alpha if use_ema else 0.0

    _print_experiment_info(args, use_compression_up or use_compression_down, is_residual)

    # Record drift for ours / ours+disable_sync
    track_drift = is_residual

    # Cold start
    # Convention: Cold start communication cost = cost of initial model distribution from server to clients.
    # - ours: If downlink compression is enabled and streaming is needed, calculate real packet size per step, broadcast to all clients.
    # - fedavg/fedprox/qsgd: Distribute "full model/quantized model" to all clients once before training starts.
    coldstart_total_bytes = 0
    coldstart_peak_bytes = 0

    if is_residual:
        init_rounds = server.init_stream_rounds()
        if init_rounds > 0 and use_compression_down:
            print(f"Cold start: Streaming initial model ({init_rounds} rounds)")
            with tqdm(range(init_rounds), desc="Cold start streaming", unit="round") as pbar:
                for r in pbar:
                    packed_delta = server.init_stream_step()

                    # Incorporate cold start traffic into cumulative communication amount
                    step_bytes = calc_comm_bytes(packed_delta)
                    step_total = step_bytes * args.num_users
                    coldstart_total_bytes += step_total
                    coldstart_peak_bytes = max(coldstart_peak_bytes, step_total)
                    total_comm_bytes += step_total

                    template = {k: torch.zeros_like(v, device='cpu')
                                for k, v in global_model.state_dict().items()}
                    dense_delta = unpack_sparse(packed_delta, template)
                    for uid in range(args.num_users):
                        clients[uid].apply_downlink(dense_delta)
                    
                    pbar.set_postfix({
                        'Comm': f"{step_bytes * args.num_users / 1024:.1f}KB",
                        'Cumulative': f"{total_comm_bytes / 1024:.1f}KB"
                    })
            print("Cold start completed")
        else:
            gw = server.get_global_weights()
            # Initial full synchronization traffic must also be counted
            coldstart_total_bytes = get_model_byte_size(gw) * args.num_users
            coldstart_peak_bytes = coldstart_total_bytes
            total_comm_bytes += coldstart_total_bytes
            for uid in range(args.num_users):
                clients[uid].set_synced_model(gw)
            print("All clients synchronized full model directly")
    else:
        # Non-residual algorithms also require this "initial model distribution" step, otherwise cumulative counts omit cold start.
        # Here according to real downlink perspective: each client receives a copy of the initial model.
        gw = server.get_global_weights()
        if is_qsgd:
            init_down_bytes_per_client = calc_qsgd_bytes(gw, args.qsgd_bits)
        else:
            init_down_bytes_per_client = get_model_byte_size(gw)
        coldstart_total_bytes = init_down_bytes_per_client * args.num_users
        coldstart_peak_bytes = coldstart_total_bytes
        total_comm_bytes += coldstart_total_bytes
        if args.verbose:
            print(
                f"[Cold Start] Initial model distribution: {init_down_bytes_per_client * args.num_users / 1024:.1f}KB, "
                f"Cumulative: {total_comm_bytes / 1024:.1f}KB"
            )

    # Explanation: Round 0 only includes downlink cold start, excluding training and uploading.
    logger.log(
        0,
        round_bytes=coldstart_total_bytes,
        cumulative_bytes=total_comm_bytes,
        peak_bytes=coldstart_peak_bytes,
        main_cumulative_bytes=0,
    )
    peak_bytes_list.append(coldstart_peak_bytes)

    # Phase 1 — Main Training Loop
    downlink_broadcast = None

    with tqdm(range(args.epochs), desc="Federated Learning Training", unit="round") as epoch_pbar:
        for epoch in epoch_pbar:
            t0 = time.time()

            m = max(int(args.frac * args.num_users), 1)

            downlink_bytes_this_round = 0

            if is_residual:
                # Apply the aggregate result of the previous round as downlink broadcast
                if downlink_broadcast is not None:
                    if use_compression_down:
                        msg_size = calc_comm_bytes(downlink_broadcast)
                    else:
                        msg_size = get_model_byte_size(downlink_broadcast)

                    # Residual algorithm requires all clients to stay synchronized, so broadcast to all.
                    # Downlink is calculated based on all users N, distinguished from uplink which only calculates selected m.
                    downlink_bytes_this_round = msg_size * args.num_users

                    # Apply downlink update to all clients
                    template = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
                    if use_compression_down:
                        dense_down = unpack_sparse(downlink_broadcast, template)
                    else:
                        dense_down = downlink_broadcast
                    for uid in range(args.num_users):
                        clients[uid].apply_downlink(dense_down)
                # else: at epoch 0 downlink_broadcast is None, no downlink operation needed
            else:
                # FedAvg / FedProx / QSGD: Round perspective adjustment
                # - From epoch>=1 onwards, issue the aggregated model from the previous round (billed to all users N).
                if epoch == 0:
                    downlink_bytes_this_round = 0
                else:
                    if is_qsgd:
                        # Fix: Calculate the volume of the quantized increment returned by the server via aggregate in the previous round
                        if downlink_broadcast is not None:
                            qsgd_delta_size = calc_qsgd_bytes(downlink_broadcast, args.qsgd_bits)
                            downlink_bytes_this_round = qsgd_delta_size * args.num_users
                        else:
                            downlink_bytes_this_round = 0
                    else:
                        model_size = get_model_byte_size(server.get_global_weights())
                        downlink_bytes_this_round = model_size * args.num_users

            total_comm_bytes += downlink_bytes_this_round

            selected = random.sample(range(args.num_users), m)
            global_weights = server.get_global_weights()

            # QSGD Downlink Quantization: Server quantizes the global model before dispatch, simulating real downlink compression noise                                                                                             
            if is_qsgd:
                # Due to DoubleSqueeze, QSGDServer maintains the global model explicitly
                # and already applies error-feedback bidding. We just fetch its model here,
                # as the compression logic from downlink is handled in QSGDServer's state,
                # or we just re-quantize the updated global weights if broadcast natively.
                # Since DoubleSqueeze broadcasts quantized weights (or differences), we can
                # just use server.get_global_weights() which represents the decompressed current state
                # simulated on the clients. For simplicity, we pass global weights directly now.
                pass

            local_results = []
            local_losses = []
            local_train_accs = [] 
            client_data_sizes = []
            uplink_bytes_this_round = 0
            epoch_init_drifts = []  # Init drift: Prior training local_model/synced_model L2 dist to global model (state mismatch)
            epoch_post_drifts = []  # Post drift: After training local model L2 dist to global model (SCAFFOLD Client Drift)

            for uid in selected:
                # Init Drift calculation
                # ours(+Sync): synced_model locked → Init Drift ≈ 0 (horizontal line)
                # (-Sync) ablation: local_model accumulates error continuously → Init Drift rises continuously (explodes)
                if track_drift:
                    if is_residual:
                        client_base = clients[uid].get_train_base_weights()
                    else:
                        client_base = None

                    if client_base is not None:
                        drift_l2 = math.sqrt(sum(
                            torch.sum((client_base[k].cpu() - global_weights[k].cpu()) ** 2).item()
                            for k in global_weights
                        ))
                        epoch_init_drifts.append(drift_l2)
                        del client_base

                result, loss, train_acc = clients[uid].local_train(
                    global_weights,       
                    epoch
                )
                local_results.append(result)
                local_losses.append(loss)
                local_train_accs.append(train_acc)  # New
                client_data_sizes.append(len(user_groups[uid]))

                # Post Drift calculation
                # ours may also face post-training deviation under Non-IID data
                if track_drift:
                    post_weights = clients[uid].get_local_weights_after_train()
                    if post_weights is not None:
                        post_drift_l2 = math.sqrt(sum(
                            torch.sum((post_weights[k].cpu() - global_weights[k].cpu()) ** 2).item()
                            for k in global_weights
                        ))
                        epoch_post_drifts.append(post_drift_l2)
                        del post_weights

                if is_residual:
                    if use_compression_up:
                        uplink_bytes_this_round += calc_comm_bytes(result)
                    else:
                        uplink_bytes_this_round += get_model_byte_size(result)
                elif is_qsgd:
                    # QSGD: Use quantized theoretical transmission amount
                    uplink_bytes_this_round += calc_qsgd_bytes(result, args.qsgd_bits)
                else:
                    # FedAvg/FedProx: Each client uploads complete state_dict
                    uplink_bytes_this_round += get_model_byte_size(result)

            total_comm_bytes += uplink_bytes_this_round
            main_cum_comm_bytes += (uplink_bytes_this_round + downlink_bytes_this_round)
            avg_loss = sum(local_losses) / len(local_losses)
            avg_train_acc = sum(local_train_accs) / len(local_train_accs)

            # Aggregation
            if is_residual:
                # Residual aggregation, return value is the downlink broadcast data for next round
                downlink_broadcast = server.aggregate(local_results, client_data_sizes)
            elif is_qsgd:
                # Update downlink_broadcast for accurate calculation in the next round
                downlink_broadcast = server.aggregate(local_results, client_data_sizes)
            else:
                server.aggregate(local_results, client_data_sizes)
                downlink_broadcast = None

            # Testing and logs
            test_acc, test_loss = server.test(test_dataset)

            # Calculate average initial drift for this round
            avg_init_drift = sum(epoch_init_drifts) / len(epoch_init_drifts) if epoch_init_drifts else 0.0
            # Calculate average post training drift for this round
            avg_post_drift = sum(epoch_post_drifts) / len(epoch_post_drifts) if epoch_post_drifts else 0.0

            round_total_bytes = uplink_bytes_this_round + downlink_bytes_this_round
            peak_bytes = max(uplink_bytes_this_round, downlink_bytes_this_round)
            peak_bytes_list.append(peak_bytes)
            log_kwargs = dict(
                test_acc=test_acc, test_loss=test_loss,
                train_loss=avg_loss, train_acc=avg_train_acc,
                round_bytes=round_total_bytes,
                cumulative_bytes=total_comm_bytes,
                peak_bytes=peak_bytes,
                main_cumulative_bytes=main_cum_comm_bytes,
            )
            if track_drift:
                log_kwargs['init_drift'] = avg_init_drift
                log_kwargs['post_drift'] = avg_post_drift
            logger.log(epoch + 1, **log_kwargs)

            if use_ema:
                if ema_acc is None:
                    ema_acc = test_acc
                else:
                    ema_acc = ema_alpha * test_acc + (1 - ema_alpha) * ema_acc

                if ema_acc > best_ema_acc:
                    best_ema_acc = ema_acc
                    patience_counter = 0
                    best_weights = server.get_global_weights()
                else:
                    patience_counter += 1
            else:
                # No EMA used, apply test_acc directly
                # Increase tolerance to avoid early stopping triggered by minor fluctuations
                if test_acc > best_ema_acc + 0.001:  # 0.1% tolerance
                    best_ema_acc = test_acc
                    patience_counter = 0
                    best_weights = server.get_global_weights()
                else:
                    patience_counter += 1

            elapsed = time.time() - t0
            
            ema_info = f", EMA: {ema_acc:.4f}" if use_ema else ""
            drift_info = f", InitDrift: {avg_init_drift:.4f}, PostDrift: {avg_post_drift:.4f}" if track_drift else ""
            progress_info = {
                'TestAcc': f"{test_acc*100:.2f}%",
                'TestLoss': f"{test_loss:.4f}",
                'TrainAcc': f"{avg_train_acc*100:.2f}%",
                'TrainLoss': f"{avg_loss:.4f}",
                'Comm': f"{round_total_bytes/1024:.1f}KB",
                'Cum': f"{total_comm_bytes/1024:.1f}KB",
                'Time': f"{elapsed:.1f}s"
            }
            if use_ema:
                progress_info['EMA'] = f"{ema_acc*100:.2f}%"
            if track_drift:
                progress_info['InitDrift'] = f"{avg_init_drift:.4f}"
                progress_info['PostDrift'] = f"{avg_post_drift:.4f}"
            
            epoch_pbar.set_postfix(progress_info)
            
            # Retain original round log output (modified to percentage format)
            epoch_pbar.write(
                f"Round {epoch+1}: "
                f"TestAcc: {test_acc*100:.2f}%, TestLoss: {test_loss:.4f}, "
                f"TrainAcc: {avg_train_acc*100:.2f}%, TrainLoss: {avg_loss:.4f}, "
                f"Comm: {round_total_bytes/1024:.1f}KB, Cum: {total_comm_bytes/1024:.1f}KB, "
                f"Time: {elapsed:.1f}s"
                f"{ema_info}{drift_info}"
            )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience: {patience})")
                break

    # Restore best weights for testing
    if best_weights is not None:
        server.global_model.load_state_dict(best_weights)

    # Final testing
    test_acc, test_loss = server.test(test_dataset)

    logger.close()

    print("Training completed")
    print(f"Total communication bytes: {total_comm_bytes / 1024:.1f} KB")
    print(f"Peak bytes in any round: {max(peak_bytes_list) / 1024:.1f} KB")
    print(f"Final test accuracy: {test_acc * 100:.2f}%, test loss: {test_loss:.4f}")
    if use_ema:
        print(f"Best EMA test accuracy: {best_ema_acc * 100:.2f}% (patience: {patience})")
    else:
        print(f"Best test accuracy: {best_ema_acc * 100:.2f}% (patience: {patience})")

# Helper print functions

def _print_experiment_info(args, use_compression, is_residual):
    """Print experimental configuration summary"""
    print(f"Experiment Setup: Algorithm={args.alg}, Dataset={args.dataset}({'IID' if args.iid else 'Non-IID'}), Model={args.model}, Epochs={args.epochs}, Clients={args.num_users}({args.frac:.0%}), Local Epochs={args.local_ep}")
    if is_residual:
        cr_up = args.cr_up if args.cr_up is not None else 'None'
        cr_down = args.cr_down if args.cr_down is not None else 'None'
        sync_status = 'Disabled(-Sync Ablation)' if getattr(args, 'disable_sync', 0) else 'Enabled(+Sync)'
        print(f"Compression Config: Uplink CR={cr_up}, Downlink CR={cr_down}, Error Feedback={'On' if getattr(args, 'use_ef', 1) else 'Off'}, Sync Replica={sync_status}")
    if args.alg == 'fedprox':
        print(f"FedProx Config: mu={args.mu}")
    if args.alg == 'qsgd':
        print(f"QSGD Config: Quantization Bits={args.qsgd_bits}")
    use_ema_str = f"EMA={'On' if args.use_ema else 'Off'}"
    device_str = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    print(f"Other Configs: {use_ema_str}, Device={device_str}, Optimizer={args.optimizer}, LR={args.lr}, Seed={args.seed}")


if __name__ == "__main__":
    main()