# BiSARC

This repository contains the official implementation of our paper recently accepted to **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**.

## Overview
This repository implements **BiSARC** (our proposed method), along with several baseline Federated Learning algorithms for comparative study. 

The implemented algorithms include:
- **FedAvg** (Federated Averaging)
- **FedProx** (Federated Proximal)
- **DoubleSqueeze** (Parallel Stochastic Gradient Descent with Double-Pass Precision-Compensated Compression)
- **FedBiF** (Baseline)
- **Ours (BiSARC)**

## Dependencies

You can install the required dependencies using `pip`:

```bash
pip install -r Code/algorithms/FedBiF_Baseline/requirements.txt
```
*(Please make sure you have appropriate PyTorch and CUDA versions based on your hardware)*

## Project Structure

```text
BiSARC/
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
    ├── Framework Consistency Test/  # Tests for consistency across frameworks
    └── main.py           # Main execution script for the framework
```

## Running Experiments

All commands should be executed inside the `Code` directory:
```bash
cd Code
```

We provide a unified entry script `main.py` for most of the algorithms. You can run different algorithms and datasets by specifying the corresponding arguments. 

### Key Arguments:
- `--alg`: The federated learning algorithm to use (`ours`, `fedavg`, `fedprox`, `DoubleSqueeze`).
- `--dataset`: The dataset to train on (`cifar` for CIFAR-10, `cifar100`, `shakespeare`, `femnist`).
- `--model`: The model architecture (`resnet18`, `cnn`, `lstm`).
- `--alpha`: The parameter for Dirichlet distribution controlling the data non-IIDness (e.g., `0.5`, `0.1`).

### Examples

**1. Running BiSARC (Ours)**
```bash
# CIFAR-10, Non-IID (alpha=0.5)
python main.py --alg ours --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --cr_up 0.2 --cr_down 0.2 --use_ef 1 --gpu 0 --seed 42
```

**2. Running Baselines (e.g., FedAvg, DoubleSqueeze)**
```bash
# FedAvg on CIFAR-100, Non-IID (alpha=0.1)
python main.py --alg fedavg --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --gpu 0 --seed 42

# DoubleSqueeze on FEMNIST
python main.py --alg doublesqueeze --dataset femnist --model cnn --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --gpu 0 --seed 42
```

**3. Running FedBiF Baseline**
The FedBiF baseline uses dedicated shell scripts located in `algorithms/FedBiF_Baseline/run/`. For example:
```bash
# CIFAR-10
bash algorithms/FedBiF_Baseline/run/fedbif3.sh 3 resnet18 0 0 0.01 8

# Shakespeare
bash algorithms/FedBiF_Baseline/run/fedbif_shakespeare.sh
```

*(Note: For a full list of commands under all different settings and baselines, please refer to the `commands.md` file.)*

## Citation

If you find this repository useful in your research, please consider citing our TNNLS paper:

```bibtex
@article{bisarc202X,
  title={...},
  author={...},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={...}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.