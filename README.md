# BiSARC

This repository contains the official implementation of our paper submitted to **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**.

## Overview

This repository implements **BiSARC** (our proposed method), along with several baseline Federated Learning algorithms for comparative study.

The implemented algorithms include:

- **FedAvg** (Federated Averaging)
- **FedProx** (Federated Proximal)
- **DoubleSqueeze** (Parallel Stochastic Gradient Descent with Double-Pass Precision-Compensated Compression)
- **FedBiF** (Baseline)
- **Ours (BiSARC)**

## Dependencies

This repo has **two** slightly different dependency sets.

### For BiSARC / FedAvg / FedProx / DoubleSqueeze

Unified entry: `Code/main.py`.

```bash
pip install -r Code/requirements.txt
```

### For FedBiF baseline

Baseline implementation lives under `Code/algorithms/FedBiF_Baseline/`.

```bash
pip install -r Code/algorithms/FedBiF_Baseline/requirements.txt
```

Notes:

- `Code/requirements.txt` intentionally does **not** pin PyTorch; please install a PyTorch/CUDA build that matches your machine.
- The FedBiF baseline requirements file pins `torch==2.2.0` / `torchvision==0.17.0` and other packages for reproducibility.

## Project Structure

```text
BiSARC-Bidirectional-State-Aligned-Compression-for-Bandwidth-Efficient-Federated-Learning/
├── README.md
├── commands.md           # Detailed command lines for different settings
└── Code/                 # Main source code directory
    ├── algorithms/       # Implementations of FL algorithms
    │   ├── FedBiF_Baseline/ # FedBiF algorithm and scripts
    │   ├── fedavg.py
    │   ├── doublesqueeze.py
    │   └── residual_fl.py
    ├── core/             # Core simulation codes for clients and servers
    ├── data/             # Data partition and sampling utilities
    ├── models/           # Neural Network models (ResNet, CNN, RNN, etc.)
    ├── utils/            # Utilities including compression, metrics, etc.
  ├── tests/             # Small tests / experiments
  │   ├── Comm Overhead Experiment/
  │   └── Framework Consistency Test/
    └── main.py           # Main execution script for the framework
```

## Running Experiments

All commands should be executed inside the `Code` directory:

```bash
cd Code
```

### Windows / PowerShell notes

- The commands in this README use `python ...` and work in PowerShell.
- The FedBiF baseline provides `.sh` scripts. On Windows you can run them via **Git Bash / WSL**, or translate the arguments to a direct `python ...` invocation.

We provide a unified entry script `main.py` for most of the algorithms. You can run different algorithms and datasets by specifying the corresponding arguments.

### Key Arguments

- `--alg`: The federated learning algorithm to use (`ours`, `fedavg`, `fedprox`, `DoubleSqueeze`).
- `--dataset`: The dataset to train on (`cifar` for CIFAR-10, `cifar100`, `shakespeare`, `femnist`).
- `--model`: The model architecture (`resnet18`, `cnn`, `lstm`).
- `--alpha`: The parameter for Dirichlet distribution controlling the data non-IIDness (e.g., `0.5`, `0.1`).

### Examples

### 1. Running BiSARC (Ours)

```bash
# CIFAR-10, Non-IID (alpha=0.5)
python main.py --alg ours --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --cr_up 0.2 --cr_down 0.2 --use_ef 1 --gpu 0 --seed 42
```

### 2. Running Baselines (e.g., FedAvg, DoubleSqueeze)

```bash
# FedAvg on CIFAR-100, Non-IID (alpha=0.1)
python main.py --alg fedavg --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --gpu 0 --seed 42

# DoubleSqueeze on FEMNIST
python main.py --alg doublesqueeze --dataset femnist --model cnn --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --gpu 0 --seed 42
```

### 3. Running FedBiF Baseline

The FedBiF baseline uses dedicated shell scripts located in `algorithms/FedBiF_Baseline/run/`. For example:

```bash
# CIFAR-10
bash algorithms/FedBiF_Baseline/run/fedbif3.sh 3 resnet18 0 0 0.01 8

# Shakespeare
bash algorithms/FedBiF_Baseline/run/fedbif_shakespeare.sh
```

If you are on Windows without `bash`, consider either:

- Using **Git Bash** / **WSL** to run the `.sh` scripts as-is, or
- Opening the corresponding `.sh` file and copying its `python ...` command into PowerShell.

*(Note: For a full list of commands under all different settings and baselines, please refer to the `commands.md` file.)*
