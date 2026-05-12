#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training metrics logging and persistence"""

import os
import csv
import matplotlib
matplotlib.use('Agg') 
matplotlib.rcParams.update({
    'font.family': ['sans-serif'],
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'axes.unicode_minus': False,  
    'figure.max_open_warning': 0  
})
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class MetricsLogger:
    """
    Log metrics like test_accuracy, train_loss, communication_cost etc. per round,
    and save them as a CSV file after training is complete.
    """

    def __init__(self, save_dir='./results'):
        self.records = []
        self.save_dir = save_dir

    def log(self, epoch, **kwargs):
        """Log metrics for one round, e.g.: log(epoch=1, test_acc=0.92, loss=0.3)"""
        row = {'epoch': epoch}
        row.update(kwargs)
        self.records.append(row)

    def save(self, args=None):
        """Save all records as CSV with experiment details txt, and plot metric curves"""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args:
            method_name = self._get_method_name(args)
            folder_name = f"{method_name}_{ts}"
        else:
            folder_name = f"experiment_{ts}"
            
        log_dir = os.path.join(self.save_dir, folder_name)
        os.makedirs(log_dir, exist_ok=True)

        csv_path = os.path.join(log_dir, f'{folder_name}.csv')
        if self.records:
            
            keys_set = set()
            for r in self.records:
                keys_set.update(r.keys())
            keys = ['epoch'] + sorted([k for k in keys_set if k != 'epoch'])
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.records)

        if args is not None:
            detail_path = os.path.join(log_dir, 'experiment_details.txt')
            with open(detail_path, 'w', encoding='utf-8') as f:
                f.write(f"Time: {ts}\n")
                for k, v in vars(args).items():
                    f.write(f"{k}: {v}\n")

        plots_path = None
        if self.records:
            plots_path = self._plot_metrics(log_dir, args)

        print(f"Logs saved to: {csv_path}")
        if plots_path:
            print(f"Plots saved to: {plots_path}")
        return csv_path

    def close(self):
        """Clean up matplotlib resources"""
        plt.close('all')

    def _plot_metrics(self, log_dir, args=None):
        """Plot training metric curves"""
        if not self.records:
            return None
        
        epochs = [r['epoch'] for r in self.records]
        train_losses = [r.get('train_loss', 0) for r in self.records]
        test_losses = [r.get('test_loss', 0) for r in self.records]
        train_accs = [r.get('train_acc', 0) for r in self.records]
        test_accs = [r.get('test_acc', 0) for r in self.records]
        client_drifts = [r.get('init_drift', None) for r in self.records]
        post_drifts = [r.get('post_drift', None) for r in self.records]
        has_drift = any(d is not None for d in client_drifts)
        has_post_drift = any(d is not None for d in post_drifts)
        
        if has_drift:
            fig, axes = plt.subplots(2, 4, figsize=(24, 10))
            (ax1, ax2, ax5, ax7), (ax3, ax4, ax6, ax8) = axes
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        fig.suptitle(f'Training Metrics - {args.alg.upper() if args else "Unknown"}', fontsize=16)
        
        # 1. Training loss curve
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Test loss curve
        ax2.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, marker='s', markersize=3)
        ax2.set_title('Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Training accuracy curve
        ax3.plot(epochs, train_accs, 'g-', label='Train Accuracy', linewidth=2, marker='^', markersize=3)
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Test accuracy curve
        ax4.plot(epochs, test_accs, 'purple', label='Test Accuracy', linewidth=2, marker='D', markersize=3)
        ax4.set_title('Test Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5, 6, 7, 8. Drift curves + Communication curve (if data available)
        if has_drift:
            # Subplot 5: Init Drift (State inconsistency, source of compression error)
            init_vals = [d if d is not None else 0 for d in client_drifts]
            ax5.plot(epochs, init_vals, 'darkorange', label='Init Drift (L2)',
                     linewidth=2, marker='x', markersize=4)
            ax5.set_title('Init Drift (Pre-Training, State Inconsistency)')
            ax5.set_xlabel('Round')
            ax5.set_ylabel('L2 Distance to Global Model')
            ax5.grid(True, alpha=0.3)
            ax5.legend()

            # Subplot 6: Post Drift (SCAFFOLD Client Drift, Non-IID data heterogeneity)
            post_vals = [d if d is not None else 0 for d in post_drifts]
            ax6.plot(epochs, post_vals, 'crimson', label='Post Drift (L2)',
                     linewidth=2, marker='+', markersize=4)
            ax6.set_title('Post Drift (Post-Training, SCAFFOLD Client Drift)')
            ax6.set_xlabel('Round')
            ax6.set_ylabel('L2 Distance to Global Model')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

            # Subplot 7: Overlaid comparison of the two drifts
            ax7.plot(epochs, init_vals, 'darkorange', label='Init Drift', linewidth=2, marker='x', markersize=3)
            ax7.plot(epochs, post_vals, 'crimson', label='Post Drift', linewidth=2, marker='+', markersize=3)
            ax7.set_title('Drift Comparison (Init vs Post)')
            ax7.set_xlabel('Round')
            ax7.set_ylabel('L2 Distance to Global Model')
            ax7.grid(True, alpha=0.3)
            ax7.legend()

            # Subplot 8: Cumulative communication
            cum_bytes = [r.get('cumulative_bytes', 0) for r in self.records]
            cum_mb = [b / (1024 * 1024) for b in cum_bytes]
            ax8.plot(epochs, cum_mb, 'teal', label='Cumulative Comm (MB)',
                     linewidth=2, marker='v', markersize=3)
            ax8.set_title('Cumulative Communication')
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('MB')
            ax8.grid(True, alpha=0.3)
            ax8.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(log_dir, 'metrics_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self._plot_comparison(log_dir, epochs, train_losses, test_losses, train_accs, test_accs, args)
        
        return plot_path

    def _plot_comparison(self, log_dir, epochs, train_losses, test_losses, train_accs, test_accs, args):
        """Plot Train vs Test comparison graph"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_title('Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, train_accs, 'g-', label='Train Accuracy', linewidth=2, marker='^', markersize=3)
        ax2.plot(epochs, test_accs, 'purple', label='Test Accuracy', linewidth=2, marker='D', markersize=3)
        ax2.set_title('Accuracy Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(f'{args.alg.upper() if args else "Unknown"} - Train vs Test Comparison', fontsize=14)
        plt.tight_layout()
        
        comparison_path = os.path.join(log_dir, 'train_vs_test_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_path

    def _get_method_name(self, args):
        """Generate method name based on args"""
        if not args:
            return "Unknown"
            
        method = args.alg
        
        iid_str = "IID" if args.iid else "NonIID"
        
        compression_parts = []
        if hasattr(args, 'cr_up') and args.cr_up is not None:
            compression_parts.append(f"UpCR{args.cr_up}")
        if hasattr(args, 'cr_down') and args.cr_down is not None:
            compression_parts.append(f"DownCR{args.cr_down}")
        
        if compression_parts:
            compression_str = "_".join(compression_parts)
            method_name = f"{method}_{iid_str}_{compression_str}"
        else:
            method_name = f"{method}_{iid_str}_NoComp"
        
        if args.alg == 'fedprox' and hasattr(args, 'mu'):
            method_name += f"_mu{args.mu}"
        if args.alg == 'qsgd' and hasattr(args, 'qsgd_bits'):
            method_name += f"_Q{args.qsgd_bits}b"
        if args.alg == 'doublesqueeze' and hasattr(args, 'doublesqueeze_bits'):
            method_name += f"_DS{args.doublesqueeze_bits}b"

        return method_name