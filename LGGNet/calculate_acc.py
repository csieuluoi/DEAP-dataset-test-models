import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# from model import LGNNet
from model_new_code import LGNNet, TSception

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from utils import get_one_subject, get_dataloader

import torchmetrics

from einops import rearrange
import torch
import torch.nn as nn
import os
import numpy as np

DEVICE = torch.device("cuda:0")
NUM_GPUS = 1
EPOCHS = 25

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


def get_final_weight(weight_dir = ""):
    file_names = os.listdir(weight_dir)

    for name in file_names:
        if name[:5] == "final":
            return name


if __name__ == "__main__":
    BATCH_SIZE = 8

    # get data of one subject
    for SUBJECT_NUMBER in range(2, 33):
        X, y, _ = get_one_subject(SUBJECT_NUMBER)

        ## implement leave one trial out
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        predictions = []
        targets = []
        accuracy = torchmetrics.Accuracy()

        y_preds = []
        y_true = []
        i = 0
        for train_index, val_index in loo.split(X):
            print("TRAIN:", train_index, "VAL:", val_index)
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            ## duplicate minority label
            # X_train = rearrange(X_train, "n c s -> n (c s)")
            # oversample = RandomOverSampler(sampling_strategy='minority')
            # X_train, y_train = oversample.fit_resample(X_train, y_train)
            # print(y_train.shape)
            # X_train = rearrange(X_train, "n (c s) -> n c s", c = 32)
            # print(X_train.shape)
            ## implement k-fold
            weight_dir = f"weights/LGGNet_subject_{SUBJECT_NUMBER:02d}/LOTO_index_{val_index[0]}"

            ##  initialize the model and trainer

            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)

            final_ckpt = get_final_weight(weight_dir)
            # print(final_ckpt)

            model = LGNNet().load_from_checkpoint(os.path.join(weight_dir, final_ckpt))
            model.to(DEVICE)
            model.eval()
            out = model(torch.Tensor(X_val).to(DEVICE))
            # model.eval()
            # out1 = model(torch.Tensor(X_val).to(DEVICE))

            # model.train()
            # out2 = model(torch.Tensor(X_val).to(DEVICE))
            # print(out, out1, out2)

            print(model.ggnn.global_A)
            # acc = accuracy(out.detach().cpu(), torch.Tensor(y_val).int())
            # print(acc)

            y_preds.append(np.argmax(out.detach().cpu().numpy()))
            y_true.append(y_val[0])
            i+=1
            if i == 5:
                break

        print(y_preds, y_true)
        print(accuracy_score(y_preds, y_true))

        break

