#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# 1. Default parameters aligned with the experiment setup.
model="resnet18"; dataset_id=3; ip_end=0; gpu_str="0"; lr=0.01; bit=8; 
momentum=0.9; l2=5e-4; epochs=5; batch_size=32; num_per_round=10; num_client=100; val_ratio=0.0; 

# External argument parsing. The original flow is kept with updated defaults.
if [ -n "$1" ]; then
    dataset_id=$1
    model=$2
    ip_end=$3
    gpu_str=$4
    lr=$5
    bit=$6
    echo "By parse: dataset-[$dataset_id] model-[$model] with ip_end-[$ip_end] on GPU-[$gpu_str] and lr-[$lr]"
else
    echo "By default: dataset-[$dataset_id] model-[$model] with ip_end-[$ip_end] on GPU-[$gpu_str] and lr-[$lr]"
fi

len=${#gpu_str}
gpu=${gpu_str:0:1}
gpu_clients=()
for ((i=0; i<$num_per_round; i++)); do
    index=$((i % len))
    gpu_clients+=("${gpu_str:$index:1}")
done

# 2. Update the global round count and partition strategy for CIFAR-10.
if [[ $dataset_id == 1 ]]; then
    dataset="fmnist"; rounds=100; ip_head="0.0.0.0:1";
    part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
elif [[ $dataset_id == 2 ]]; then
    dataset="svhn"; rounds=100; ip_head="0.0.0.0:2";
    part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
elif [[ $dataset_id == 3 ]]; then
    dataset="cifar10"; rounds=400; ip_head="0.0.0.0:3"; # Use 400 global rounds.
    part_strategy_list=("labeldir0.5")                  # Run only Dirichlet alpha=0.5 Non-IID.
elif [[ $dataset_id == 4 ]]; then
    dataset="cifar100"; rounds=200; ip_head="0.0.0.0:4";
    part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
elif [[ $dataset_id == 5 ]]; then
    dataset="tinyimagenet"; rounds=200; ip_head="0.0.0.0:5";
    part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
else
    echo "wrong dataset."
    exit 1
fi
#######################################################################################

com_type="fedbif"; ip_mid=7; 
ip="${ip_head}${ip_mid}${ip_end}"

# 3. Run only the configured labeldir0.5 setting (index 0).
for b in 0; do  
    part_strategy=${part_strategy_list[${b}]}

    dir="../log/${dataset}+${num_client}/${dataset}+${part_strategy}/${model}+${com_type}+bit_${bit}+lr_${lr}/"
    mkdir -p ${dir} # Ensure the log directory exists.

    python ../train/server.py \
        --com_type ${com_type} --bit ${bit} \
        --model ${model} --dataset ${dataset} --lr ${lr} --momentum ${momentum} --l2 ${l2} \
        --rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} \
        --num_per_round ${num_per_round} --num_client ${num_client} \
        --gpu ${gpu} --ip ${ip} --log_dir ${dir} &

    for a in $(seq 0 $(($num_per_round-1))); do # client
    python ../train/client.py \
        --com_type ${com_type} \
        --model ${model} --dataset ${dataset} --part_strategy ${part_strategy} --num_client ${num_client} --id ${a} --val_ratio ${val_ratio} \
        --gpu ${gpu_clients[${a}]} --ip ${ip} --log_dir ${dir} &
    done
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM 
    wait
done
