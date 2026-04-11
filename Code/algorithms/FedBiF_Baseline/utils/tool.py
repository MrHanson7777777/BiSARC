import os
import json
import torch
import socket
import numpy as np
from datetime import datetime


def get_device(gpu):
    device= torch.device('cpu')
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpu))
    return device

def save_record(npy_path, record, name=None):
    if name == None:
        name = "record"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    server_name = socket.gethostname()
    filename = f"{name}_{server_name}_{timestamp}.json"
    full_path = os.path.join(npy_path, filename)
    with open(full_path, 'w') as f:
        json.dump(record, f, indent=4)

def write_avg_json(log_dir, name="record", tags=[]):
    pt_path = os.path.join(log_dir, "results")
    data_list = []
    files = []
    for filename in os.listdir(pt_path):
        if filename.startswith(name) and filename.endswith(".json"):
            file_path = os.path.join(pt_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            data_list.append(data)
            files.append(filename)

    all_records = {}
    for tag in tags:
        all_records[tag] = [float(d[tag]) for d in data_list]
    
    results = {}
    results['avg'] = {k: float(np.array(v).mean()) for k, v in all_records.items()}
    results['std'] = {k: float(np.array(v).std()) for k, v in all_records.items()}
    results['all'] = all_records
    results['all']['files'] = files
    json_path = os.path.join(log_dir, name+".json")
    with open(json_path, 'w+') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    print("done")
