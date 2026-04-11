#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

###############################################################################
# FedBIF on FEMNIST (LEAF natural non-IID partition)
###############################################################################

# ---- Default hyper-parameters ----
model="cnn";         # cnn (CNN_MNIST 62-class) or resnet18
gpu_str="0";         # GPU ids, e.g. "0" or "01"
lr=0.01;
bit=8;
momentum=0.9;
l2=5e-4;
epochs=5;
batch_size=32;
num_per_round=10;    # clients sampled per round
num_client=100;      # total FL clients (<=370)
val_ratio=0.0;
rounds=400;

# ---- Parse optional arguments ----
if [ -n "$1" ]; then
    model=$1
    gpu_str=$2
    lr=$3
    bit=$4
    rounds=$5
    echo "By parse: model-[$model] GPU-[$gpu_str] lr-[$lr] bit-[$bit] rounds-[$rounds]"
else
    echo "By default: model-[$model] GPU-[$gpu_str] lr-[$lr] bit-[$bit] rounds-[$rounds]"
fi

len=${#gpu_str}
gpu=${gpu_str:0:1}
gpu_clients=()
for ((i=0; i<$num_per_round; i++)); do
    index=$((i % len))
    gpu_clients+=("${gpu_str:$index:1}")
done

dataset="femnist"
part_strategy="natural"
ip_head="0.0.0.0:6"
ip_mid=7
ip_end=0
ip="${ip_head}${ip_mid}${ip_end}"

###############################################################################
# 1. Generate FEMNIST partition (only first time)
###############################################################################
echo ">>> Generating FEMNIST partition npy ..."
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from dataloaders.femnist import generate_femnist_partition
for n in [100, 200, 370]:
    try:
        generate_femnist_partition(num_clients=n, seed=1234)
    except ValueError as e:
        print(f'Skipping {n}: {e}')
"

###############################################################################
# 2. Start FedBIF
###############################################################################
com_type="fedbif"

dir="../log/${dataset}+${num_client}/${dataset}+${part_strategy}/${model}+${com_type}+bit_${bit}+lr_${lr}/"
mkdir -p ${dir}

echo ">>> Starting FedBIF on FEMNIST: ${model}, ${rounds} rounds, bit=${bit}, lr=${lr}"

python ../train/server.py \
    --com_type ${com_type} --bit ${bit} \
    --model ${model} --dataset ${dataset} --lr ${lr} --momentum ${momentum} --l2 ${l2} \
    --rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} \
    --num_per_round ${num_per_round} --num_client ${num_client} \
    --gpu ${gpu} --ip ${ip} --log_dir ${dir} &

for a in $(seq 0 $(($num_per_round-1))); do
python ../train/client.py \
    --com_type ${com_type} \
    --model ${model} --dataset ${dataset} --part_strategy ${part_strategy} --num_client ${num_client} --id ${a} --val_ratio ${val_ratio} \
    --gpu ${gpu_clients[${a}]} --ip ${ip} --log_dir ${dir} &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait

echo ">>> FedBIF on FEMNIST done!"
