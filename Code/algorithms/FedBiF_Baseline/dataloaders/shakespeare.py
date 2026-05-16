"""
Shakespeare Dataset Loader for Federated Learning (LEAF-style).

Data source: The Complete Works of William Shakespeare (Project Gutenberg)
Task: Next-character prediction (character-level language model)
Partition: Natural Non-IID by play-character (same as LEAF benchmark)

Each FL client corresponds to one role (e.g., "HAMLET_HAMLET"),
whose lines form a naturally non-IID text corpus.
"""

import collections
import io
import os
import re
import sys
import json
import glob
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset

# Vocabulary (same as LEAF Shakespeare)
ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[]abcdefghijklmnopqrstuvwxyz}"
)
NUM_LETTERS = len(ALL_LETTERS)   # 80

# Map char to index and index to char.
CHAR2IDX = {c: i for i, c in enumerate(ALL_LETTERS)}
IDX2CHAR = {i: c for i, c in enumerate(ALL_LETTERS)}

SEQ_LEN = 80   # input sequence length (same as LEAF)

# Paths

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TXT_PATH     = os.path.join(_ROOT, "dataset", "input.txt")
SHAKESPEARE_DIR  = os.path.dirname(RAW_TXT_PATH)
PROCESSED_JSON   = os.path.join(SHAKESPEARE_DIR, "shakespeare_all_data.json")

# Official LEAF source URL: pg100.txt (UTF-8 plain text).
# Ref: LEAF preprocess_shakespeare.py comment:
#   http://www.gutenberg.org/cache/epub/100/pg100.txt
PG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
# Fallback zip (LEAF get_data.sh originally uses this 1994 zip):
PG_ZIP_URL  = "http://www.gutenberg.org/files/100/old/1994-01-100.zip"
PG_ZIP_NAME = "100.txt"


# Download

def download_raw(url: str = PG_URL, dest: str = RAW_TXT_PATH) -> str:
    """
    Use local input.txt as Shakespeare data. Do not download.
    """
    if os.path.exists(dest):
        print(f"[Shakespeare] Local text found: {dest}")
        return dest
    raise FileNotFoundError(f"Shakespeare input file not found: {dest}")


# Preprocessing

def _clean_char(c: str) -> str:
    """Map a character to a vocab character, or return '' to drop it."""
    return c if c in CHAR2IDX else ''


def _text_to_indices(text: str):
    """Filter text to vocab chars and return list of indices."""
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def _remove_nonalphanumerics(s: str) -> str:
    return re.sub(r'\W+', '_', s)


def preprocess_shakespeare(
    raw_path: str = RAW_TXT_PATH,
    out_path: str = PROCESSED_JSON,
    min_samples: int = 5,
    seq_len: int = SEQ_LEN,
) -> dict:
    """
    Parse raw Shakespeare text (pg100.txt, LEAF format) and build per-user sequences.

    Follows LEAF's preprocess_shakespeare.py logic exactly:
      Character line:  "^  ([a-zA-Z][a-zA-Z ]*)\\. (.*)"
        -> 2 spaces indent, name (mixed case), period, space, first dialogue on SAME line
      Continuation:    "^    (.*)"
        -> 4 spaces indent
      Play detection:  line contains "by William Shakespeare"
        -> title is 2-7 lines above that marker
      Comedy of Errors: uses different regex (no indent)

    Each user key: _remove_nonalphanumerics(play_title + '_' + CHARACTER_NAME)
    x[i]: list of seq_len character indices (ALL_LETTERS vocab)
    y[i]: index of the next character
    """
    if os.path.exists(out_path):
        print(f"[Shakespeare] Preprocessed data already exists: {out_path}")
        with open(out_path, 'r') as f:
            return json.load(f)


    print("[Shakespeare] Preprocessing raw text (input.txt custom format) ...")
    with open(raw_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    user_texts = {}
    current_role = None
    buffer = []
    for line in lines + ['\n']:
        line = line.rstrip('\r\n')
        if line.endswith(':') and len(line) > 1:
            if current_role and buffer:
                text = ' '.join(buffer).replace('\n', ' ')
                text = re.sub(r'   *', ' ', text)
                if current_role in user_texts:
                    user_texts[current_role] += ' ' + text
                else:
                    user_texts[current_role] = text
                buffer = []
            current_role = line[:-1].strip()
        elif line.strip() == '':
            continue
        else:
            if current_role:
                buffer.append(line.strip())
    if current_role and buffer:
        text = ' '.join(buffer).replace('\n', ' ')
        text = re.sub(r'   *', ' ', text)
        if current_role in user_texts:
            user_texts[current_role] += ' ' + text
        else:
            user_texts[current_role] = text

    print(f"[Shakespeare] Parsed {len(user_texts)} roles.")

    # Build (x, y) sequence pairs.
    users, num_samples_list = [], []
    user_data = {}


    for role, text in user_texts.items():
        indices = _text_to_indices(text)
        if len(indices) < seq_len + min_samples:
            continue
        xs, ys = [], []
        for i in range(len(indices) - seq_len):
            xs.append(indices[i: i + seq_len])
            ys.append(indices[i + seq_len])
        if len(xs) < min_samples:
            continue
        users.append(role)
        num_samples_list.append(len(xs))
        user_data[role] = {'x': xs, 'y': ys}

    data = {'users': users, 'num_samples': num_samples_list, 'user_data': user_data}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f)

    print(f"[Shakespeare] Preprocessed {len(users)} users, "
          f"total samples: {sum(num_samples_list)}")
    print(f"[Shakespeare] Saved to {out_path}")
    return data


# Load

def load_shakespeare_data(
    json_path: str = PROCESSED_JSON,
    train_ratio: float = 0.9,
    seed: int = 1234,
):
    """
    Load processed Shakespeare JSON and split train/test per user.

    Returns
    -------
    train_data       : {'x': list[list[int]], 'y': list[int]}
    test_data        : {'x': list[list[int]], 'y': list[int]}
    user_train_map   : {client_id (int): [indices in train_data]}
    user_names       : list of user id strings
    """
    if not os.path.exists(json_path):
        # Try to build from scratch
        download_raw()
        preprocess_shakespeare()

    with open(json_path, 'r') as f:
        raw = json.load(f)

    np.random.seed(seed)

    all_train_x, all_train_y = [], []
    all_test_x,  all_test_y  = [], []
    user_train_map = {}
    user_names = []

    train_idx = 0
    for user in raw['users']:
        xs = raw['user_data'][user]['x']
        ys = raw['user_data'][user]['y']
        n  = len(xs)

        idx = np.random.permutation(n)
        split = int(n * train_ratio)
        tr_idx = idx[:split]
        te_idx = idx[split:]

        cid = len(user_train_map)
        user_names.append(user)
        user_train_map[cid] = list(range(train_idx, train_idx + len(tr_idx)))
        train_idx += len(tr_idx)

        for i in tr_idx:
            all_train_x.append(xs[i])
            all_train_y.append(ys[i])
        for i in te_idx:
            all_test_x.append(xs[i])
            all_test_y.append(ys[i])

    train_data = {'x': all_train_x, 'y': all_train_y}
    test_data  = {'x': all_test_x,  'y': all_test_y}

    print(f"[Shakespeare] Loaded {len(user_names)} users, "
          f"train: {len(all_train_x)}, test: {len(all_test_x)}")
    return train_data, test_data, user_train_map, user_names


# PyTorch Dataset

class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for Shakespeare next-character prediction.
    Each item: (x_tensor LongTensor[seq_len], y_scalar LongTensor)
    """
    def __init__(self, data: dict):
        self.x = data['x']   # list of lists of ints
        self.y = data['y']   # list of ints

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# Partition generation

def generate_shakespeare_partition(
    num_clients: int = 100,
    seed: int = 1234,
    save_dir: str = None,
) -> dict:
    """
    Generate npy partition file for Shakespeare (natural LEAF partition by user).
    Selects `num_clients` users randomly from all available users.
    """
    _, _, user_train_map, user_names = load_shakespeare_data(seed=seed)

    total_users = len(user_names)
    if num_clients > total_users:
        raise ValueError(
            f"Requested {num_clients} clients but only {total_users} "
            f"Shakespeare users available."
        )

    np.random.seed(seed)
    selected = sorted(np.random.choice(total_users, num_clients, replace=False))

    net_dataidx_map = {}
    for new_id, old_id in enumerate(selected):
        net_dataidx_map[new_id] = user_train_map[old_id]

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_npy_")
    os.makedirs(save_dir, exist_ok=True)

    npy_name = f"shakespeare-natural-{num_clients}-{seed}.npy"
    npy_path = os.path.join(save_dir, npy_name)
    np.save(npy_path, net_dataidx_map)
    print(f"[Shakespeare] Partition saved to {npy_path}")
    return net_dataidx_map


# Global cache

_shakespeare_cache: dict = {}

def get_shakespeare_train_data(seed: int = 1234):
    if seed not in _shakespeare_cache:
        train_data, test_data, user_train_map, user_names = load_shakespeare_data(seed=seed)
        _shakespeare_cache[seed] = {
            'train_data':     train_data,
            'test_data':      test_data,
            'user_train_map': user_train_map,
            'user_names':     user_names,
        }
    return _shakespeare_cache[seed]


# CLI

if __name__ == "__main__":
    # 1. Download raw text
    download_raw()
    # 2. Preprocess
    preprocess_shakespeare()
    # 3. Generate partition for 100 clients
    generate_shakespeare_partition(num_clients=100, seed=1234)
