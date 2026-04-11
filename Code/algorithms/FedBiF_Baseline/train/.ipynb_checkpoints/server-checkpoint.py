import os
import sys
import json
os.environ["NO_PROXY"] = "*"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
import random
import socket
import warnings
import argparse
from datetime import datetime
sys.path.append('.')
sys.path.append('..')
warnings.filterwarnings("ignore")
from collections import OrderedDict
import torch
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

from train.trainer import test
from utils.logger import get_log
from utils.tool import get_device, save_record, write_avg_json
from dataloaders.dataloader import get_server_test_dataloader

from plugin.models import get_model_with_config
from plugin.bif import bif_before_server_send

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="server")
    parser.add_argument("--bit", type=float, default=4, help="fedbif")     # dadaquant max_bit
    parser.add_argument("--com_type", type=str, default="fedavg", help="communication type")
    parser.add_argument("--model", type=str, default="resnet18", help="model")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--lr", type=float, default=0.1, help="lr")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--l2", type=float, default=0.0, help="l2")
    parser.add_argument("--rounds", type=int, default=100, help="rounds")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--save_best", type=int, default=0, help="save_round")
    parser.add_argument("--save_round", type=int, default=0, help="save_round")
    parser.add_argument("--num_per_round", type=int, default=10, choices=range(2, 400), help="num_per_round")
    parser.add_argument("--num_client", type=int, default=100, choices=range(2, 400), help="num_client")
    parser.add_argument("--gpu", type=int, default=4, help="-1 0 1")
    parser.add_argument("--ip", type=str, default="0.0.0.0:12345", help="server address")
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    args = parser.parse_args()

    ## logger
    server_name = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = get_log(args.log_dir, f"server_{server_name}_{timestamp}")
    device = get_device(args.gpu)
    logger.info(f"server_name: {server_name}")
    for key, value in vars(args).items(): 
        logger.info(f"{key}: {value}")
    pt_path = os.path.join(args.log_dir, "results")
    os.makedirs(pt_path, exist_ok=True)
    
    ## model
    model_config = {
        "com_type": args.com_type,
        }
    logger.info(model_config)
    model, keys_weight = get_model_with_config(args.model, args.dataset, model_config)
    model.to(device)
    print(model)

    ## fl prepare
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    record = {"accuracy": [], "loss": []}
    com_dict = {
        "ids": "0.1.2.3.4.5.6.7.8.9", 
        "seed": 1234, 
        }

    def fit_com_dict(com_dict):
        ids = random.sample(list(range(args.num_client)), args.num_per_round)
        com_dict["ids"] = '.'.join(map(str, ids))
        com_dict["seed"] = random.randint(a=0, b=2024)

    def fit_config(server_round: int):
        config = {
            "com_type": args.com_type,
            "round": server_round,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "l2": args.l2,

            "ids": com_dict["ids"],
            "seed": com_dict["seed"],
        }
        return config

    def get_evaluate_fn(model, dataset):
        test_loader = get_server_test_dataloader(dataset, batch_size=args.batch_size)

        def evaluate(server_round, parameters, config):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            if server_round == 0:
                print("skipping server evalutation...")
                loss, accuracy = 0.0, 0.0
            else:
                print("starting server evalutation...")
                loss, accuracy = test(model, test_loader, None, device)
                record["accuracy"].append(accuracy)
                record["loss"].append(loss)
                logger.info("round %d - server test loss:%.4f; acc:%.4f" % (server_round, loss, accuracy))
                if args.save_round:
                    torch.save(model.state_dict(), os.path.join(pt_path, str(server_round) + ".pt"))
                if args.save_best and accuracy >= np.max(np.array(record["accuracy"])):
                    torch.save(model.state_dict(), os.path.join(pt_path, "best.pt"))

            
            if args.com_type == "fedbif":
                next_bit_pos = int(args.bit-1-server_round%args.bit)
                bif_before_server_send(model, args.bit, next_bit_pos)
                logger.info(f"Next round bit_pos: {next_bit_pos}")
                torch.save(model.state_dict(), os.path.join(pt_path, str(server_round) + ".pt"))
                file_path = os.path.join(pt_path, f"{server_round-1}.pt")
                if os.path.exists(file_path):
                    os.remove(file_path)

            fit_com_dict(com_dict)  # prepare ids and seeds for next round
            return loss, {"accuracy": accuracy}

        return evaluate

    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_per_round,
        min_evaluate_clients=0,
        min_available_clients=args.num_per_round,
        evaluate_fn=get_evaluate_fn(model, args.dataset),
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )
    for i in range(args.num_per_round):
        np.save(os.path.join(pt_path, "flags_" + str(i) + ".npy"), [])

    fl.server.start_server(
        server_address=args.ip,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    json_tags = ["best_accuracy", "best_round"]
    best_round = np.argmax(np.array(record["accuracy"]))
    best_acc = record["accuracy"][best_round]
    best_loss = record["loss"][best_round]
    record["best_accuracy"] = best_acc
    record["best_round"] = int(best_round)
    logger.info("* best round: %d; best acc: %.4f; best loss: %.4f" % (best_round, best_acc, best_loss))
    save_record(pt_path, record, name="record")
    write_avg_json(args.log_dir, name="record", tags = json_tags)
