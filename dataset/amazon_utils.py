import json
from collections import defaultdict
from typing import List, Dict
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import math
from torchtext.data.utils import get_tokenizer
from text_generation import sample_by_ratings


TRAIN_TEST_RATIO = 0.8


def read_path(path):
    data_dct = defaultdict(list)
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    with open(path, 'r') as inf:
        for line in tqdm(inf.readlines(), desc=f"Path:{path}"):
            item = json.loads(line)
            reviewerID, reviewText, score = item['reviewerID'], item['reviewText'], item['overall']
            if len(reviewText) < 2 or len(tokenizer(reviewText)) > 30:
                continue
            if score == 5.0 and np.random.rand() < 0.5:
                continue
            data_dct[reviewerID].append([reviewText, score])
    return data_dct


def merge_data_by_category(dcts: List[Dict]):
    full_dct = defaultdict(list)
    for dct in dcts:
        for key, value in dct.items():
            full_dct[key].extend(value)
    to_delete = []
    for key, value in full_dct.items():
        if len(value) < 16:
            to_delete.append(key)
    for key in to_delete:
        full_dct.pop(key)
    return full_dct


def inspect_data_statistics(full_dct: Dict):
    status = [0, 0, 0, 0, 0]
    for key, value in full_dct.items():
        for entry in value:
            score = int(entry[1]) - 1
            status[score] += 1
    # global STATUS
    # STATUS = status
    print(status)
    print(f">=3: {sum(status[3:])}, <3: {sum(status[:3])}, fraction: {sum(status[3:]) / sum(status)}")
    # exit()

def check_label_distribution(labels):
    result = [labels.count(i+1) for i in range(5)]
    return result


def split_dataset(full_dct, train_size=TRAIN_TEST_RATIO):
    train_dct, test_dct = {}, {}
    for client_id, data_pairs in full_dct.items():
        train, test = train_test_split(data_pairs, train_size=train_size, random_state=100)
        train_dct[client_id] = train
        test_dct[client_id] = test

    return train_dct, test_dct


def find_labels_to_transfer(labels, ratio: float = 1):
    target_num = math.ceil(max([labels.count(i+1) for i in range(5)]) * ratio)
    to_generate = []
    for i in range(1, 6):
        to_generate.extend([i] * max(target_num - labels.count(i), 0))
    return to_generate

def find_labels_to_transfer_binary(labels, ratio = 1):
    neg_num = labels.count(0) + labels.count(1) + labels.count(3)
    pos_num = labels.count(4) + labels.count(5)
    difference = pos_num - neg_num
    difference = math.floor(difference * ratio)
    if difference == 0:
        return []
    if difference > 0:
        to_generate = list(np.random.choice([1, 2, 3], difference, replace=True, p=[0.4, 0.5, 0.1]))
    else:
        to_generate = list(np.random.choice([4, 5], -difference, replace=True, p=[0.6, 0.4]))
    # print(to_generate)
    return to_generate

def make_client_dataset(data_pairs, text_pipeline, label_pipeline, usage="tune", augment=False):
    texts = [item[0] for item in data_pairs]
    labels = [item[1] for item in data_pairs]

    # if usage == "tune" and augment:
    #     to_generate = find_labels_to_transfer_binary(labels)
    #     generated = sample_by_ratings(to_generate)
    #     texts.extend(generated)
    #     labels.extend(to_generate)

    dataset = MyDataset(texts, labels)
    return dataset

def find_texts_to_transfer(data_pairs, label_pipeline):
    labels = [item[1] for item in data_pairs]
    return find_labels_to_transfer(labels, 1)

    neg_num = labels.count(0) + labels.count(1) + labels.count(3)
    pos_num = labels.count(4) + labels.count(5)
    difference = pos_num - neg_num
    if difference == 0:
        return []
    if difference > 0:
        to_generate = list(np.random.choice([1, 2, 3], difference, replace=True, p=[0.25, 0.5, 0.25]))
    else:
        to_generate = list(np.random.choice([4, 5], -difference, replace=True, p=[0.6, 0.4]))
    return to_generate

def incorporate_augment_and_make_client_dataset(data_pairs, text_pipeline, label_pipeline, augment_data):
    texts = [item[0] for item in data_pairs] + augment_data[0]
    labels = [item[1] for item in data_pairs] + augment_data[1]
    dataset = MyDataset(texts, labels)
    return dataset






def read_dir(root_dir="dataset/amazon_reviews/data_by_user_small.npy"):
    # print("Reading Data...")
    # files = os.listdir(root_dir)
    # files = [f for f in files if f.endswith('.json')]
    # category_dcts = [read_path(os.path.join(root_dir, file)) for file in files if file.split("_")[1] == 'Books']
    # print("Merging...")
    # full_dct = merge_data_by_category(category_dcts)
    # # output_full_dataset(full_dct, "amazon_reviews/data_for_transfer/source.txt",
    # #                     "amazon_reviews/data_for_transfer/target.txt", "amazon_reviews/data_for_transfer/full.txt")
    # np.save("amazon_reviews/data_by_user_books.npy", full_dct)
    #
    full_dct = np.load(root_dir, allow_pickle=True).item()
    assert isinstance(full_dct, dict)
    print(f"Total number of clients: {len(full_dct.keys())}")
    inspect_data_statistics(full_dct)
    print("Splitting...")
    train_dct, test_dct = split_dataset(full_dct)
    # output_train_test_set(train_dct, test_dct, "amazon_reviews/data_for_transfer/sentiment.train",
    #                       "amazon_reviews/data_for_transfer/sentiment.test", "amazon_reviews/data_for_transfer/source.txt",
    #                       "amazon_reviews/data_for_transfer/target.txt", "amazon_reviews/data_for_transfer/full.txt")
    del full_dct
    return train_dct, test_dct


def output_train_test_set(train_dct, test_dct, train_path, test_path, src_path, tgt_path, full_path):
    """
    1 for positive, 0 for negative.
    :param src_path:
    :param train_dct:
    :param test_dct:
    :param train_path:
    :param test_path:
    :return:
    """
    train_file_0 = open(train_path + '.0', 'w')
    train_file_1 = open(train_path + ".1", 'w')
    test_file_0 = open(test_path + '.0', 'w')
    test_file_1 = open(test_path + '.1', 'w')
    src_file = open(src_path, 'w')
    tgt_file = open(tgt_path, 'w')
    full_file = open(full_path, 'w')

    def output_to_file(dct, file_1, file_0):
        for user, samples in tqdm(dct.items()):
            for entry in samples:
                label = int(entry[1])
                tokens = [token.text for token in nlp(entry[0])]
                if label > 3:
                    file_1.write(" ".join(tokens).strip() + "\n")
                    src_file.write(" ".join(tokens).strip() + "\n")
                else:
                    file_0.write(" ".join(tokens).strip() + "\n")
                    tgt_file.write(" ".join(tokens).strip() + "\n")
                full_file.write(" ".join(tokens).strip() + "\n")

        file_1.close()
        file_0.close()

    output_to_file(train_dct, train_file_1, train_file_0)
    output_to_file(test_dct, test_file_1, test_file_0)
    src_file.close()
    tgt_file.close()
    full_file.close()


def output_full_dataset(full_dct, src_path, tgt_path, full_path):
    src_file = open(src_path, 'w')
    tgt_file = open(tgt_path, 'w')
    full_file = open(full_path, 'w')

    for user, samples in tqdm(full_dct.items()):
        for entry in samples:
            label = int(entry[1])
            if label > 3:
                src_file.write(" ".join([token.text for token in nlp(entry[0])]) + "\n")
            else:
                tgt_file.write(" ".join([token.text for token in nlp(entry[0])]) + "\n")
            full_file.write(" ".join([token.text for token in nlp(entry[0])]) + "\n")

def load_full_dataset(path, augment=False, ratio=0.5, binary=True):
    full_dct = np.load(path, allow_pickle=True).item()
    assert isinstance(full_dct, dict)
    train_dct, test_dct = split_dataset(full_dct)
    train_texts, test_texts, train_labels, test_labels = [], [], [], []
    for samples in train_dct.values():
        train_texts += [item[0] for item in samples]
        train_labels += [item[1] for item in samples]
    for samples in test_dct.values():
        test_texts += [item[0] for item in samples]
        test_labels += [item[1] for item in samples]

    if ratio < 1:
        train_texts, train_labels, deleted_labels = inspect_and_filter_dataset(train_texts, train_labels, ratio, binary=binary)
    del train_dct, test_dct
    if augment:
        print("Augmenting...")
        # augment_labels = find_labels_to_transfer_binary(train_labels, ratio=3)
        # augment_labels = find_labels_to_transfer(train_labels, ratio=1)
        # augment_labels = train_labels
        augment_labels = deleted_labels
        generated = sample_by_ratings(augment_labels)
        train_texts += generated
        train_labels += augment_labels
    print(f"Train labels: {check_label_distribution(train_labels)}")
    print(f"Test labels: {check_label_distribution(test_labels)}")
    train_dataset = MyDataset(train_texts, train_labels)
    test_dataset = MyDataset(test_texts, test_labels)
    return train_dataset, test_dataset

def inspect_and_filter_dataset(texts, labels, ratio=0.5, binary=True):
    np_texts = np.array(texts)
    np_labels = np.array(labels)
    if binary:
        positive_labels_index = np_labels > 3
        negative_labels_index = np_labels <= 3
        positive_labels, positive_texts = np_labels[positive_labels_index], np_texts[positive_labels_index]
        negative_labels, negative_texts = np_labels[negative_labels_index], np_texts[negative_labels_index]
    else:
        label_5_index = np_labels == 5
        other_labels_index = np_labels != 5
        positive_labels, positive_texts = np_labels[label_5_index], np_texts[label_5_index]
        negative_labels, negative_texts = np_labels[other_labels_index], np_texts[other_labels_index]
    sampled_indexes = np.random.choice(np.arange(len(positive_labels)), math.floor(len(positive_labels) * ratio), replace=False)
    chosen_texts = positive_texts[sampled_indexes]
    chosen_labels = positive_labels[sampled_indexes]
    deleted_indexes = np.array([i for i in range(len(positive_labels)) if i not in sampled_indexes])
    deleted_labels = positive_labels[deleted_indexes]
    new_texts = np.append(negative_texts, chosen_texts)
    new_labels = np.append(negative_labels, chosen_labels)

    return new_texts.tolist(), new_labels.tolist(), deleted_labels.tolist()


class MyDataset(Dataset):
    def __init__(self, texts, labels):
        super(MyDataset, self).__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

if __name__ == "__main__":
    read_dir("amazon_reviews")
