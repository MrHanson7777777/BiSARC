# 通信开销对比实验（可单独运行）

对应脚本：`Code/comm_overhead_experiment.py`

该脚本会对比 3 个算法的通信开销，并输出两张图：

- **Figure 1**：Single-round peak bandwidth requirement across communication rounds（包含 t=0 初始化阶段）
- **Figure 2**：Cumulative communication volume during the main training phase（不含 t=0，仅主训练阶段累计）

算法与设定：

- `ours`：上下行压缩率 `cr_up=0.1`、`cr_down=0.1`（下发给所有客户端）
- `doublesqueeze`：4-bit 随机量化（带 error feedback），下发给所有客户端
- `fedavg`：32-bit dense，下发给所有客户端

## 依赖

仓库里已有 `Code/algorithms/FedBiF_Baseline/requirements.txt`，但为便于在云环境快速安装，这里额外提供了一个更小的：`Code/requirements.txt`（不含 PyTorch）。

你需要单独安装与 CUDA 匹配的 PyTorch，然后再安装其他依赖：

```powershell
# 进入代码目录
cd Code

# 先按你的 CUDA / Driver 版本安装 torch（示例命令仅供参考）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 再装其余依赖
pip install -r requirements.txt
```

## 运行

在 `Code/` 目录下运行（参数和 `main.py` 一致，比如 dataset/model/epochs 等）：

```powershell
cd Code
python comm_overhead_experiment.py --dataset mnist --model cnn --epochs 50 --num_users 100 --frac 0.1 --local_ep 5 --iid 1 --gpu 0
```

如果你需要用 **ResNet18** 做测试（更推荐在 CIFAR/CIFAR100 上）：

```bash
cd Code
python comm_overhead_experiment.py --dataset cifar --model resnet18 --epochs 50 --num_users 100 --frac 0.1 --local_ep 5 --iid 0 --alpha 0.5 --lr 0.01 --lr_scheduler cosine --gpu 0
```

输出图片默认存到：`Code/outputs/comm_overhead/`

同时会导出一份逐轮通信统计：`Code/outputs/comm_overhead/comm_overhead.csv`

## 快速自检（Smoke Test）

想先确认脚本能跑通并产出图（但不想花很久训练）可以用：

```powershell
cd Code
python comm_overhead_experiment.py --dataset mnist --model cnn --gpu 0
```

如果你在 Linux / Docker 容器（bash）里运行，请用：

```bash
cd Code
export BISARC_SMOKE=1
python comm_overhead_experiment.py --dataset mnist --model cnn --gpu 0
```

如果你要跑你命令里指定的完整轮次（例如 `--epochs 50`），请先关闭该环境变量：

```bash
unset BISARC_SMOKE
```

该模式会自动把轮数/客户端数/本地 epoch 缩小到一个很小的规模，只用于连通性验证。
