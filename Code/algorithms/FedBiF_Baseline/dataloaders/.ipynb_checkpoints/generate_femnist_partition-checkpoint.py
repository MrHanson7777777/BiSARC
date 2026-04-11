"""
Generate partition npy files for FEMNIST and Shakespeare datasets.
Run this script once before starting any FL experiment.

Usage:
    python dataloaders/generate_femnist_partition.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.femnist import generate_femnist_partition
from dataloaders.shakespeare import (
    download_raw,
    preprocess_shakespeare,
    generate_shakespeare_partition,
)

if __name__ == "__main__":
    seed = 1234

    # ── FEMNIST ──────────────────────────────────────────────────────────────
    print("=== FEMNIST ===")
    try:
        for num_clients in [100, 200, 370]:
            try:
                generate_femnist_partition(num_clients=num_clients, seed=seed)
            except ValueError as e:
                print(f"Skipping num_clients={num_clients}: {e}")
    except FileNotFoundError as e:
        print(f"[FEMNIST] Skipped (data not found): {e}")

    # ── Shakespeare ───────────────────────────────────────────────────────────
    print("\n=== Shakespeare ===")
    download_raw()
    preprocess_shakespeare()
    for num_clients in [100]:
        try:
            generate_shakespeare_partition(num_clients=num_clients, seed=seed)
        except ValueError as e:
            print(f"Skipping num_clients={num_clients}: {e}")
