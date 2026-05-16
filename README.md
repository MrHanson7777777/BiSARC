# BiSARC

This repository contains the implementation of **BiSARC: Bidirectional State-Aligned Residual Compression for Bandwidth-Efficient Federated Learning**, together with several federated learning baselines used for comparison.

## Implemented Methods

- **BiSARC** (`--alg bisarc`)
- **FedAvg** (`--alg fedavg`)
- **FedProx** (`--alg fedprox`)
- **QSGD** (`--alg qsgd`)
- **DoubleSqueeze** (`--alg doublesqueeze`)
- **FedBiF baseline** (`Code/algorithms/FedBiF_Baseline/`)

## Repository Layout

```text
.
├── README.md
├── commands.md
└── Code/
    ├── main.py
    ├── requirements.txt
    ├── algorithms/
    ├── core/
    ├── data/
    ├── models/
    ├── utils/
    └── tests/
        ├── Comm Overhead Experiment/
        └── Framework Consistency Test/
```

## Environment

Install PyTorch separately so that it matches your CUDA or CPU environment. Then install the project dependencies:

```bash
cd Code
pip install -r requirements.txt
```

For the FedBiF baseline, use its dedicated dependency file:

```bash
pip install -r algorithms/FedBiF_Baseline/requirements.txt
```

## Command Reference

The complete training command list is maintained in [`commands.md`](commands.md). Treat that file as the canonical command-line reference for reproducing the main accuracy experiments. Commands in `commands.md` should be run from the `Code/` directory:

```bash
cd Code
```

## Accuracy Experiments

Accuracy experiments use the unified simulator entry:

```bash
python main.py [arguments]
```

The ready-to-run commands for CIFAR-10, CIFAR-100, Shakespeare, FEMNIST, and the main baselines are listed in [`commands.md`](commands.md). For example:

```bash
python main.py --alg bisarc --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --cr_up 0.2 --cr_down 0.2 --use_ef 1 --gpu 0 --seed 42
```

For baseline accuracy experiments, copy the corresponding `fedavg`, `fedprox`, `qsgd`, or `doublesqueeze` command from [`commands.md`](commands.md) and run it from `Code/`.

FedBiF uses dedicated scripts:

```bash
bash algorithms/FedBiF_Baseline/run/fedbif3.sh 3 resnet18 0 0 0.01 8
bash algorithms/FedBiF_Baseline/run/fedbif_shakespeare.sh
```

On Windows, run the FedBiF `.sh` files with Git Bash or WSL.

## Communication Overhead Experiment

The communication-overhead experiment is a standalone script under `Code/tests/Comm Overhead Experiment/`. It compares `bisarc`, `fedavg`, and `doublesqueeze`, then writes two figures and a CSV report.

Run it from `Code/`:

```bash
python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset cifar --model resnet18 --epochs 50 --num_users 100 --frac 0.1 --local_ep 5 --iid 0 --alpha 0.5 --lr 0.01 --lr_scheduler cosine --gpu 0
```

Outputs are saved to:

```text
Code/outputs/comm_overhead/
```

For a short smoke test:

```bash
BISARC_SMOKE=1 python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset mnist --model cnn --gpu 0
```

PowerShell equivalent:

```powershell
$env:BISARC_SMOKE="1"
python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset mnist --model cnn --gpu 0
Remove-Item Env:BISARC_SMOKE
```

More details are in [`Code/tests/Comm Overhead Experiment/README_comm_overhead.md`](Code/tests/Comm%20Overhead%20Experiment/README_comm_overhead.md).

## Encoding/Decoding Time Overhead

The time-overhead benchmarks are the two standalone files in `Code/tests/Framework Consistency Test/`. They do not define extra command-line flags; run each file directly.

PyTorch benchmark:

```bash
cd Code
python "tests/Framework Consistency Test/PyTorch.py"
```

TensorFlow benchmark:

```bash
cd Code
python "tests/Framework Consistency Test/TensorFlow.py"
```

Both benchmarks require a GPU environment. The PyTorch benchmark writes:

```text
Code/pytorch_benchmark_results_corrected.csv
Code/latency_breakdown_corrected.png
```

The TensorFlow benchmark writes:

```text
Code/tf_benchmark_results_corrected.csv
Code/tf_latency_breakdown_corrected.png
```

## Main Arguments

- `--alg`: `bisarc`, `fedavg`, `fedprox`, `qsgd`, or `doublesqueeze`
- `--dataset`: `mnist`, `cifar`, `cifar100`, `femnist`, or `shakespeare`
- `--model`: `cnn`, `resnet18`, or `lstm`
- `--iid`: `1` for IID data and `0` for non-IID data
- `--alpha`: Dirichlet non-IID parameter, such as `0.5` or `0.1`
- `--cr_up`, `--cr_down`: BiSARC uplink/downlink compression ratios
- `--qsgd_bits`: QSGD quantization bits
- `--doublesqueeze_bits`: DoubleSqueeze quantization bits
