import torch
import os
import numpy as np
from torch.utils.data import Dataset
from utils import load_DEAP

# class DEAP_Fnet_dataset(Dataset):
#     def __init__(self, data_dir, label_type = [0, 2], transform = None):
#         self.data, self.labels = load_DEAP(data_dir,  label_type = label_type)
#         self.labels = torch.Tensor(self.labels).long()
#         print(self.labels)
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.labels[index]

#         # y = torch.Tensor(y).long()
#         # print(y)
#         if self.transform:
#             x = self.transform(x)
#         x = torch.Tensor(x).float()
#         return (x, y)

class DEAP_Fnet_dataset(Dataset):
    def __init__(self, data, labels, transform = None):
        self.data, self.labels = data, labels
        self.labels = torch.Tensor(self.labels).long()
        # print(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        # y = torch.Tensor(y).float()
        # print(y)
        if self.transform:
            x = self.transform(x)
        # x = torch.Tensor(x).float()
        return (x, y)
