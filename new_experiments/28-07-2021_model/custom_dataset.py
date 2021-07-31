import torch
import os
import numpy as np
from torch.utils.data import Dataset
from preprocessing import dataset_prepare

class DEAP_dataset(Dataset):
    def __init__(self, data, labels, transform = None):
        self.data, self.labels =  data, labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        # print("x shape: ", x.shape)
        # print("y shape: ", y.shape, y)

        if self.transform:
            x = self.transform(x)

        x = torch.Tensor(x) # transform to torch tensor
        y = torch.Tensor(y).long().squeeze()
        return (x, y)

# class DEAP_Fnet_dataset(Dataset):
#     def __init__(self, data, labels, transform = None):
#         self.data, self.labels = data, labels
#         self.labels = torch.Tensor(self.labels).long()
#         # print(self.labels)
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.labels[index]

#         # y = torch.Tensor(y).float()
#         # print(y)
#         if self.transform:
#             x = self.transform(x)
#         # x = torch.Tensor(x).float()
#         return (x, y)
