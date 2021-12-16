import sys
import os
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
from torch.utils import data
import json

class DatasetGenerator(data.Dataset):
    def __init__(self, index, data, label):
        self.index = index
        self.data=data
        self.label=label
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return np.array(self.index[index]).astype(np.float32), np.array(self.data[index]).astype(np.float32), np.array(self.label[index]).astype(np.float32)

def read_dataset(paths):
    dataset_index = []
    dataset_dataset = []
    dataset_label = []
    for path in paths:
        with open(path, "r") as temp_file:
            temp_json = json.load(temp_file)
        temp_index, temp_dataset, temp_label = temp_json
        dataset_index.extend(temp_index)
        dataset_dataset.extend(temp_dataset)
        dataset_label.extend(temp_label)
    return DatasetGenerator(dataset_index, dataset_dataset, dataset_label)

if __name__ == "__main__":
    dev_h_dataset = read_dataset(["../dataset/RACE_BERT_mini/dev_h/data.json", "../dataset/RACE_BERT_mini/dev_m/data.json"])