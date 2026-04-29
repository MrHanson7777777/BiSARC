# CIFAR-10

## $\alpha$ = 0.5

python main.py --alg fedavg --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --gpu 0 --seed 42

python main.py --alg fedprox --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --gpu 0 --seed 42

python main.py --alg ours --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --cr_up 0.2 --cr_down 0.2 --use_ef 1 --gpu 0 --seed 42

python main.py --alg doublesqueeze --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --doublesqueeze_bits 4 --gpu 0 --seed 42

## $\alpha$ = 0.1

python main.py --alg fedavg --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --gpu 0 --seed 42

python main.py --alg fedprox --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --gpu 0 --seed 42

python main.py --alg ours --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --cr_up 0.2 --cr_down 0.2 --use_ef 1 --gpu 0 --seed 42

python main.py --alg doublesqueeze --dataset cifar --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --doublesqueeze_bits 4 --gpu 0 --seed 42

# CIFAR-100

## $\alpha$ = 0.5

python main.py --alg fedavg --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --gpu 0 --seed 42

python main.py --alg fedprox --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --gpu 0 --seed 42

python main.py --alg ours --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --cr_up 0.1 --cr_down 0.1 --use_ef 1 --gpu 0 --seed 42

python main.py --alg doublesqueeze --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.5 --doublesqueeze_bits 4 --gpu 0 --seed 42

## $\alpha$ = 0.1

python main.py --alg fedavg --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --gpu 0 --seed 42

python main.py --alg fedprox --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --gpu 0 --seed 42

python main.py --alg ours --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --cr_up 0.1 --cr_down 0.1 --use_ef 1 --gpu 0 --seed 42

python main.py --alg doublesqueeze --dataset cifar100 --model resnet18 --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --alpha 0.1 --doublesqueeze_bits 4 --gpu 0 --seed 42

# Shakespeare

python main.py \
    --alg fedavg \
    --dataset shakespeare \
    --model lstm \
    --iid 0 \
    --epochs 400 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 2 \
    --local_bs 16 \
    --lr 0.8 \
    --seq_len 80 \
    --gpu 0 \
    --seed 42

python main.py \
    --alg fedprox \
    --dataset shakespeare \
    --model lstm \
    --iid 0 \
    --epochs 400 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 2 \
    --local_bs 16 \
    --lr 0.8 \
    --seq_len 80 \
    --gpu 0 \
    --seed 42

python main.py \
    --alg ours \
    --dataset shakespeare \
    --model lstm \
    --iid 0 \
    --epochs 400 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 2 \
    --local_bs 16 \
    --lr 0.8 \
    --seq_len 80 \
    --cr_up 0.1 \
    --cr_down 0.1 \
    --use_ef 1 \
    --gpu 0 \
    --seed 42

python main.py \
    --alg doublesqueeze \
    --dataset shakespeare \
    --model lstm \
    --iid 0 \
    --epochs 400 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 2 \
    --local_bs 16 \
    --lr 0.8 \
    --doublesqueeze_bits 8 \
    --seq_len 80 \
    --gpu 0 \
    --seed 42

# FEMNIST

python main.py --alg fedavg --dataset femnist --model cnn --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --gpu 0 --seed 42

python main.py --alg fedprox --dataset femnist --model cnn --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --gpu 0 --seed 42

python main.py --alg ours --dataset femnist --model cnn --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --cr_up 0.1 --cr_down 0.1 --use_ef 1 --gpu 0 --seed 42

python main.py --alg doublesqueeze --dataset femnist --model cnn --epochs 400 --num_users 100 --frac 0.1 --local_ep 5 --lr 0.01 --lr_scheduler cosine --iid 0 --gpu 0 --seed 42

---
**Note:** For the execution commands and scripts related to the **FedBiF** baseline algorithm, please refer to the `README.md` file.
