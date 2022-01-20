import json
import os
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from augment import cGAN_DataGenerator

def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in tqdm(files, desc="Loading Data..."):
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir, only_digits=True):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
            keys: client_ids(e.g. f2249_75)
            values: a dict of the form {x: [num_samples, 28*28], y: [num_samples, ]}
        test_data: dictionary of test data
    '''
    train_clients, train_data = read_dir(train_data_dir)
    _, test_data = read_dir(test_data_dir)

    if only_digits:
        remove_clients = set()
        for key in train_clients:
            train_labels = train_data[key]['y']
            test_labels = test_data[key]['y']
            if all([label >= 10 for label in train_labels]) or all([label >= 10 for label in test_labels]):
                del train_data[key]
                del test_data[key]
                remove_clients.add(key)
        train_clients = list(set(train_clients).difference(remove_clients))

    return train_clients, train_data, test_data


def save_image_dataset(dataset, path):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for images, labels in dataloader:
        torch.save(images, path + "/x.pt")
        torch.save(labels, path + "/y.pt")
        break


def load_dataset_from_saved_file(path, shape=(-1, 1, 28, 28)):
    images = torch.load(path + "/x.pt").float().reshape(shape)
    labels = torch.load(path + "/y.pt")
    return TensorDataset(images, labels)


def combine_femnist_data(train_data_dir, only_digits=True, path='tmp'):
    if os.path.exists(path + '/x.pt'):
        return load_dataset_from_saved_file(path)
    train_clients, train_data = read_dir(train_data_dir)
    images = torch.tensor([], dtype=torch.float)
    labels = torch.tensor([], dtype=torch.int)
    total = len(train_data)
    while train_data:
        if (total - len(train_data) + 1) % 100 == 0:
            print(f"Combining {total - len(train_data) + 1}/{total} user's data...")
        data = train_data.popitem()[1]
        x, y = torch.FloatTensor(data['x']), torch.LongTensor(data['y'])
        del data
        if only_digits:
            selected = y < 10
            x, y = x[selected], y[selected]
        x = (0.5 - x) / 0.5
        images = torch.cat([images, x])
        labels = torch.cat([labels, y])
    assert len(images) == len(labels)
    images = images.reshape(-1, 1, 28, 28)
    dataset = TensorDataset(images, labels)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        save_image_dataset(dataset, path)
    return dataset


def make_torch_dataset(data={'x': [], 'y': []}, only_digits=False, usage="train", augment=True, image_max_num=128,
                       threshold=0.8):
    num_classes = 10 if only_digits else 62
    x, y = torch.FloatTensor(data['x']), torch.LongTensor(data['y'])
    if only_digits:
        selected = y < 10
        x, y = x[selected], y[selected]
    x = (0.5 - x) / 0.5

    if augment:
        compliment_labels = find_compliment_labels(y, num_classes)
        if usage == "global":
            if len(compliment_labels) != 0:
                compliment_images, compliment_labels = cGAN_DataGenerator.generate_by_labels(compliment_labels,
                                                                                             threshold=threshold)
                x = torch.cat([x, compliment_images])
                y = torch.cat([y, compliment_labels])

                # if usage == "tune":
                #     if len(compliment_labels) != 0:
                #         compliment_images, compliment_labels = cGAN_DataGenerator.generate_by_labels(compliment_labels,
                #                                                                                  threshold=threshold)
                x = torch.cat([x, compliment_images])
                y = torch.cat([y, compliment_labels])

    dataset = TensorDataset(x, y)
    return dataset


def find_compliment_labels(labels: torch.Tensor, label_num=10):
    label_statistics = torch.bincount(labels, minlength=label_num)
    target_num_labels = torch.max(label_statistics)
    compliment_labels = []
    for i in range(label_num):
        augment_array_per_label = [i] * max(target_num_labels - label_statistics[i], 0)
        compliment_labels.extend(augment_array_per_label)
    return torch.LongTensor(compliment_labels)


if __name__ == "__main__":
    combine_femnist_data("femnist/data/train", False, "femnist_full/train")
    combine_femnist_data("femnist/data/train", True, "femnist_digits/train")
    # a = find_compliment_labels(torch.ones(10))
    # print(a)
