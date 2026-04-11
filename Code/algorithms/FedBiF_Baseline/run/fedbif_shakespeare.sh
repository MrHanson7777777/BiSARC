#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

###############################################################################
# FedBIF on Shakespeare (LEAF natural Non-IID partition by role/character)
# Task: next-character prediction (character-level language model)
###############################################################################

# ---- Default hyper-parameters ----
model="lstm";        # only lstm supported for Shakespeare
gpu_str="0";
lr=0.8;              # higher lr works better for LSTM char-level LM
bit=8;
momentum=0.0;        # no momentum for LSTM by default
l2=0.0;
epochs=5;
batch_size=32;
num_per_round=10;
num_client=100;
val_ratio=0.0;
rounds=400;

# ---- Parse optional arguments ----
if [ -n "$1" ]; then
    gpu_str=$1
    lr=$2
    bit=$3
    rounds=$4
    echo "By parse: GPU-[$gpu_str] lr-[$lr] bit-[$bit] rounds-[$rounds]"
else
    echo "By default: GPU-[$gpu_str] lr-[$lr] bit-[$bit] rounds-[$rounds]"
fi

len=${#gpu_str}
gpu=${gpu_str:0:1}
gpu_clients=()
for ((i=0; i<$num_per_round; i++)); do
    index=$((i % len))
    gpu_clients+=("${gpu_str:$index:1}")
done

dataset="shakespeare"
part_strategy="natural"
ip_head="0.0.0.0:8"
ip_mid=7
ip_end=0
ip="${ip_head}${ip_mid}${ip_end}"

###############################################################################
# 1. Download + preprocess + generate partition (idempotent)
###############################################################################
# Remove stale cache files so we re-download/re-preprocess with the correct URL & parser
RAW_TXT="../dataset/leaf_shakespeare/shakespeare_raw.txt"
PROC_JSON="../dataset/leaf_shakespeare/shakespeare_all_data.json"
if [ -f "$RAW_TXT" ];   then rm -f "$RAW_TXT";   echo "Removed stale $RAW_TXT";   fi
if [ -f "$PROC_JSON" ]; then rm -f "$PROC_JSON"; echo "Removed stale $PROC_JSON"; fi

echo ">>> Preparing Shakespeare dataset and partition npy ..."
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from dataloaders.shakespeare import download_raw, preprocess_shakespeare, generate_shakespeare_partition
download_raw()
preprocess_shakespeare()
generate_shakespeare_partition(num_clients=100, seed=1234)
"

###############################################################################
# 2. Start FedBIF
###############################################################################
com_type="fedbif"

dir="../log/${dataset}+${num_client}/${dataset}+${part_strategy}/${model}+${com_type}+bit_${bit}+lr_${lr}/"
mkdir -p ${dir}

echo ">>> Starting FedBIF on Shakespeare: ${model}, ${rounds} rounds, bit=${bit}, lr=${lr}"

python ../train/server.py \
    --com_type ${com_type} --bit ${bit} \
    --model ${model} --dataset ${dataset} --lr ${lr} --momentum ${momentum} --l2 ${l2} \
    --rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} \
    --num_per_round ${num_per_round} --num_client ${num_client} \
    --gpu ${gpu} --ip ${ip} --log_dir ${dir} &

for a in $(seq 0 $(($num_per_round-1))); do
python ../train/client.py \
    --com_type ${com_type} \
    --model ${model} --dataset ${dataset} --part_strategy ${part_strategy} \
    --num_client ${num_client} --id ${a} --val_ratio ${val_ratio} \
    --gpu ${gpu_clients[${a}]} --ip ${ip} --log_dir ${dir} &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait

echo ">>> FedBIF on Shakespeare done!"
