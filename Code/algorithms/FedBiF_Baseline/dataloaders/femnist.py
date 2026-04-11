"""
FEMNIST Dataset Loader for LEAF format.
Loads FEMNIST data from LEAF JSON files and provides PyTorch Dataset interfaces.
"""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


FEMNIST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "leaf_femnist", "leaf-master", "data", "femnist", "data", "sampled_data"
)

FEMNIST_NUM_CLASSES = 62  # 10 digits + 26 lowercase + 26 uppercase


def load_femnist_data(data_dir=None, train_ratio=0.9, seed=1234):
    """
    Load all FEMNIST JSON files from LEAF and split into train/test per user.
    
    Returns:
        train_data: dict with keys 'x' (list of images), 'y' (list of labels)
        test_data: dict with keys 'x', 'y'
        user_train_map: dict mapping client_id -> list of indices in train_data
    """
    if data_dir is None:
        data_dir = FEMNIST_DATA_DIR

    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No JSON files found in {data_dir}. "
            "Please ensure FEMNIST LEAF data is placed correctly."
        )

    np.random.seed(seed)

    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []
    user_train_map = {}  # client_id -> [indices in train set]
    user_names = []

    train_idx = 0

    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        for user in data['users']:
            user_names.append(user)
            x = data['user_data'][user]['x']
            y = data['user_data'][user]['y']
            n = len(x)

            # Shuffle per user then split
            indices = np.random.permutation(n)
            split = int(n * train_ratio)
            train_indices = indices[:split]
            test_indices = indices[split:]

            client_id = len(user_train_map)
            user_train_map[client_id] = list(range(train_idx, train_idx + len(train_indices)))
            train_idx += len(train_indices)

            for i in train_indices:
                all_train_x.append(x[i])
                all_train_y.append(y[i])
            for i in test_indices:
                all_test_x.append(x[i])
                all_test_y.append(y[i])

    train_data = {'x': all_train_x, 'y': all_train_y}
    test_data = {'x': all_test_x, 'y': all_test_y}

    print(f"[FEMNIST] Loaded {len(user_names)} users, "
          f"train samples: {len(all_train_x)}, test samples: {len(all_test_x)}")

    return train_data, test_data, user_train_map, user_names


class FemnistDataset(Dataset):
    """PyTorch Dataset for FEMNIST."""

    def __init__(self, data, transform=None):
        """
        Args:
            data: dict with keys 'x' (list of 784-dim pixel vectors) and 'y' (list of labels)
            transform: optional torchvision transforms
        """
        self.x = data['x']
        self.y = data['y']
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Convert 784-dim vector to 28x28 image tensor
        img = np.array(self.x[idx], dtype=np.float32).reshape(28, 28)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        label = int(self.y[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


def generate_femnist_partition(num_clients=100, seed=1234, save_dir=None):
    """
    Generate npy partition file for FEMNIST based on natural LEAF user split.
    If num_clients <= total users, select num_clients users.
    If num_clients > total users, raise error.
    """
    _, _, user_train_map, user_names = load_femnist_data(seed=seed)

    total_users = len(user_names)
    if num_clients > total_users:
        raise ValueError(
            f"Requested {num_clients} clients but only {total_users} FEMNIST users available."
        )

    np.random.seed(seed)
    selected_users = np.random.choice(total_users, num_clients, replace=False)
    selected_users = sorted(selected_users)

    net_dataidx_map = {}
    for new_id, old_id in enumerate(selected_users):
        net_dataidx_map[new_id] = user_train_map[old_id]

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_npy_")
    os.makedirs(save_dir, exist_ok=True)

    npy_name = f"femnist-natural-{num_clients}-{seed}.npy"
    npy_path = os.path.join(save_dir, npy_name)
    np.save(npy_path, net_dataidx_map)
    print(f"[FEMNIST] Partition saved to {npy_path}")

    return net_dataidx_map


# Global cache to avoid re-loading data multiple times
_femnist_cache = {}


def get_femnist_train_data(seed=1234):
    """Get cached FEMNIST training dataset."""
    if seed not in _femnist_cache:
        train_data, test_data, user_train_map, user_names = load_femnist_data(seed=seed)
        _femnist_cache[seed] = {
            'train_data': train_data,
            'test_data': test_data,
            'user_train_map': user_train_map,
            'user_names': user_names,
        }
    return _femnist_cache[seed]


if __name__ == "__main__":
    # Generate partition files
    seed = 1234
    for num_clients in [100, 200]:
        generate_femnist_partition(num_clients=num_clients, seed=seed)
