import json
import os
import glob

data_dir = 'dataset/leaf_femnist/leaf-master/data/femnist/data/sampled_data/'
files = sorted(glob.glob(os.path.join(data_dir, '*.json')))

all_users = []
all_labels = set()
total_samples = 0

for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    all_users.extend(data['users'])
    for u in data['users']:
        total_samples += len(data['user_data'][u]['y'])
        all_labels.update(data['user_data'][u]['y'])

print(f'Total JSON files: {len(files)}')
print(f'Total users: {len(all_users)}')
print(f'Total samples: {total_samples}')
print(f'Label range: {min(all_labels)} - {max(all_labels)}, count: {len(all_labels)}')
print(f'Samples per user (min/max): {total_samples//len(all_users)}')
