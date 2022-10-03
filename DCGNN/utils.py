import pandas as pd
import pickle
import numpy as np
import time
import os

from torch.utils.data import TensorDataset, DataLoader
import torch

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def load_data_per_subject(i):
    subject_path = (
        "D:\AIproject\emotion recognition\DEAP\data_preprocessed_python/"
        + f"s{i:02n}.dat"
    )
    subject = pickle.load(open(subject_path, "rb"), encoding="latin1")

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

        labels_arousal[
            (1 <= new_labels[:, 1]) & (new_labels[:, 1] <= median_arousal)
        ] = 0
        labels_arousal[
            (median_arousal < new_labels[:, 1]) & (new_labels[:, 1] <= 9)
        ] = 1

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
        labels_val[
            (low_value < new_labels[:, 0]) & (new_labels[:, 0] <= high_value)
        ] = 1
        labels_val[(high_value < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 2

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= low_value)] = 0
        labels_arousal[
            (low_value < new_labels[:, 1]) & (new_labels[:, 1] <= high_value)
        ] = 1
        labels_arousal[(high_value < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 2
    #         new_labels[:, 0] = labels_val
    #         new_labels[:, 1] = labels_arousal
    #         output_labels = new_labels[:, 0:2]
    output_labels = [labels_val, labels_arousal]

    return np.array(output_labels)


def get_one_subject(subject_index, num_classes=2):
    # get data of one subject according to the subject's index (1 to 32)
    # subject_index: an interger in range (1, 32)
    # return:
    #   data (numpy array),
    #   val_labels: binary label of valence,
    #   ar_label: binary label of arousal.

    data = load_data_per_subject(subject_index)
    X = data["data"][:, :32, :]
    labels = data["labels"]
    assert num_classes in [2, 3], "num_classes should be 2 or 3"
    labels_k_classes = labels_quantization(labels, num_classes)

    val_labels_k = labels_k_classes[0]
    ar_labels_k = labels_k_classes[1]

    return X, val_labels_k, ar_labels_k


def get_dataloader(X, y, batch_size=1, num_workers=1, shuffle=True):

    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(y).long().squeeze()
    # tensor_y = tensor_y.unsqueeze(1)
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    my_dataloader = DataLoader(
        my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )  # create your dataload

    return my_dataloader


### helper classes
class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return "{:.1f}h".format(x / 3600)
        if x >= 60:
            return "{}m".format(round(x / 60))
        return "{}s".format(x)


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = x
    print("using gpu:", x)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def normalize(train, test):
    """
    this function do standard normalization for EEG channel by channel
    :param train: training data
    :param test: testing data
    :return: normalized training and testing data
    """
    # data: sample x 1 x channel x data
    mean = 0
    std = 0
    for channel in range(train.shape[2]):
        mean = np.mean(train[:, :, channel, :])
        std = np.std(train[:, :, channel, :])
        train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
        test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
    return train, test


if __name__ == "__main__":
    X, val_labels, ar_labels = get_one_subject(1)

    print(X.shape)
    print(val_labels)
