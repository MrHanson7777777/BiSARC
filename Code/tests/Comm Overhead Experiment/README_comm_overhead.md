# Communication Overhead Experiment

This standalone experiment compares the communication overhead of three algorithms:

- `bisarc`: bidirectional residual Top-K compression with `cr_up=0.1` and `cr_down=0.1`
- `doublesqueeze`: 4-bit stochastic quantization with error feedback
- `fedavg`: dense 32-bit model/update transmission

The script writes two figures and one CSV file:

- `figure1_peak_bandwidth.png`: single-round peak bandwidth requirement, including the `t=0` cold-start initialization.
- `figure2_cumulative_main_phase.png`: cumulative communication volume during the main training phase, excluding `t=0`.
- `comm_overhead.csv`: per-round communication statistics for all compared algorithms.

## Setup

Install PyTorch separately for your CUDA or CPU environment, then install the remaining dependencies:

```bash
cd Code
pip install -r requirements.txt
```

## Run

Run the script from the `Code/` directory. It accepts the same dataset, model, and training arguments as `main.py`.

```bash
cd Code
python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset mnist --model cnn --epochs 50 --num_users 100 --frac 0.1 --local_ep 5 --iid 1 --gpu 0
```

For CIFAR or CIFAR-100 with ResNet-18:

```bash
cd Code
python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset cifar --model resnet18 --epochs 50 --num_users 100 --frac 0.1 --local_ep 5 --iid 0 --alpha 0.5 --lr 0.01 --lr_scheduler cosine --gpu 0
```

Outputs are saved to:

```text
Code/outputs/comm_overhead/
```

## Smoke Test

For a short connectivity check, set `BISARC_SMOKE=1`. The script will shrink the default number of rounds, clients, and local epochs unless you explicitly override those arguments.

PowerShell:

```powershell
cd Code
$env:BISARC_SMOKE="1"
python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset mnist --model cnn --gpu 0
Remove-Item Env:BISARC_SMOKE
```

Bash:

```bash
cd Code
export BISARC_SMOKE=1
python "tests/Comm Overhead Experiment/comm_overhead_experiment.py" --dataset mnist --model cnn --gpu 0
unset BISARC_SMOKE
```
