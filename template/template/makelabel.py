# -*- coding: utf-8 -*-

import csv

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

def get_label(label_path: str, phones: list) -> list:
    labels = []
    with open(label_path, mode="r") as f_reader:
        reader = csv.reader(f_reader)
        l = [row for row in reader]
    listmax = max([(len(x[0].split(" "))) for x in l])
    for ele in l:
        label = []
        ele = ele[0].split(" ")
        for index, phone in enumerate(ele):
            if phone in phones:
                data = phones.index(phone)
            else:
                data = phones.index("?")
                print("?")
            label.append(data)
        labels.append(F.pad(torch.tensor(label), (0, listmax - len(label)), "constant", phones.index("_")))
    return labels

def get_phones(label_path: str) -> list:

    with open(label_path, mode="r") as f_reader:
        reader = f_reader.readlines()
        line_list = []
        for row in reader:
            line = row.strip("\n").split(" ")
            line_list += line

    phones = list(set(line_list))
    phones.insert(0, "_")
    phones.insert(1, "<s>")
    phones.insert(2, "</s>")
    phones.append("|")
    phones.append("?")
    return phones

def get_phones_csv(label_path: str) -> list:
    print(label_path)
    train_data = pd.read_csv(label_path)
    train_label = train_data["preprocessed_hiragana"].tolist()
    # train_label = train_data["original"].tolist()
    char_storage = []
    for sentence in train_label:
        for char in sentence:
            char_storage.append(char)
    phones = list(set(char_storage))
    phones.insert(0, "_")
    phones.insert(1, "<s>")
    phones.insert(2, "</s>")
    phones.append("|")
    phones.append("?")
    return phones

def get_label_csv(label_path: str, phones: list) -> list:
    labels = []
    read_data = pd.read_csv(label_path)
    label = read_data["preprocessed_hiragana"].tolist()
    # label = read_data["original"].tolist()
    listmax = max([len(x) for x in label])
    for ele in tqdm(label):
        label = []
        for index, phone in enumerate(ele):
            if phone in phones:
                data = phones.index(phone)
            else:
                data = phones.index("?")
                print("?")
            label.append(data)
        labels.append(F.pad(torch.tensor(label), (0, listmax - len(label)), "constant", phones.index("_")))
    return labels

if __name__ == "__main__":
    phones = get_phones("assets/train_label.txt")
    labels = get_label("assets/train_label.txt", phones)
