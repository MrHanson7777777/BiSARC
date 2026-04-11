# FedBiF

**FedBiF: Communication-efficient Federated Learning via Bits Freezing [TPDS 2025]**

## Environment Setup
Please install the necessary dependencies first:
```
pip install -r requirements.txt
```

## Data Partition
Please run the following code to download and partition datasets:
```
python ./dataloaders/datapartition.py 
```

## Run Experiments
Please use the scripts to run the experiments, for example:
```
./run/fedavg.sh
./run/fedbif.sh
```

## Citation
```
@article{li2025fedbif,
  title={FedBiF: Communication-efficient Federated Learning via Bits Freezing},
  author={Shiwei Li and Qunwei Li and Haozhao Wang and Ruixuan Li and Jianbin Lin and Wenliang Zhong},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={xx},
  number={xx},
  pages={xxxx--xxxx},
  year={2025},
  publisher={IEEE}
}
```