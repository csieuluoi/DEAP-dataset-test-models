import pandas as pd
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
def load_data_per_subject(i):
    subject_path = "D:\AIproject\emotion recognition\DEAP\data_preprocessed_python/"+f"s{i:02n}.dat"
    subject = pickle.load(open(subject_path, 'rb'), encoding = 'latin1')

    return subject


def labels_quantization(labels, num_classes):
    new_labels = labels
    if num_classes == 2:
    #     median_val = np.median(labels[:, 0])
    #     print(median_val)

    #     median_arousal = np.median(labels[:, 1])
    #     print(median_arousal)

        median_val = 5
        median_arousal = 5

        labels_val = np.zeros(new_labels.shape[0])
        labels_arousal = np.zeros(new_labels.shape[0])

        labels_val[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= median_val)] = 0
        labels_val[(median_val < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 1

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= median_arousal)] = 0
        labels_arousal[(median_arousal < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 1

#         new_labels[:, 0] = labels_val
#         new_labels[:, 1] = labels_arousal
#         output_labels = new_labels[:, 0:2]

    elif num_classes == 3:
    #     median_val = np.median(labels[:, 0])

    #     median_arousal = np.median(labels[:, 1])

        low_value = 4
        high_value = 6

        labels_val = np.zeros(new_labels.shape[0])
        labels_arousal = np.zeros(new_labels.shape[0])

        labels_val[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= low_value)] = 0
        labels_val[(low_value < new_labels[:, 0]) & (new_labels[:, 0] <= high_value)] = 1
        labels_val[(high_value < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 2

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= low_value)] = 0
        labels_arousal[(low_value < new_labels[:, 1]) & (new_labels[:, 1] <= high_value)] = 1
        labels_arousal[(high_value < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 2
#         new_labels[:, 0] = labels_val
#         new_labels[:, 1] = labels_arousal
#         output_labels = new_labels[:, 0:2]
    output_labels = [labels_val, labels_arousal]

    return np.array(output_labels)

def get_one_subject(subject_index):
    data = load_data_per_subject(subject_index)
    X = data["data"][:, :32, 128*3:]
    labels = data["labels"]

    labels_2_classes = labels_quantization(labels,2)
    val_labels_2 = labels_2_classes[0]
    ar_labels_2 = labels_2_classes[1]

    return X, val_labels_2, ar_labels_2


def get_dataloader(X, y, batch_size = 1, shuffle = True):

    tensor_x = torch.Tensor(X) # transform to torch tensor
    tensor_y = torch.Tensor(y)
    tensor_y = tensor_y.unsqueeze(1)
    my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size = batch_size, shuffle = shuffle) # create your dataload


    return my_dataloader

if __name__=="__main__":
    X, val_labels, ar_labels = get_one_subject(1)

    print(X.shape)
    print(val_labels)
