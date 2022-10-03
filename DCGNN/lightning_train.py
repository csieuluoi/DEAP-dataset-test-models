from turtle import window_height
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import torch_geometric.nn as tgnn

# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from utils import get_dataloader, get_one_subject
from data_preprocess import extract_one_subject_features, extract_all_features
import numpy as np

from GraphCN import GraphCN
from spectralGNN import SpectralGNN
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import time
import os

from datetime import datetime

def device_as(x, y):
    return x.to(y.device)


# tensor operations now support batched inputs
def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    size = a.shape[-1]
    a += device_as(torch.eye(size), a)
    D_norm = calc_degree_matrix_norm(a)
    L_norm = torch.bmm(torch.bmm(D_norm, a), D_norm)
    return L_norm


class LitModel(pl.LightningModule):
    def __init__(
        self,
        in_features,
        out_features,
        n_channels,
        K,
        n_classes=2,
        dropout_rate=0.0,
        learning_rate=1e-1, # in the paper they said they search this value in [0.1, 1]
        weight_decay= 1e-4,#1e-4,
        lambda_ = 0.3 # searched in [0.1, 1]
    ):
        super().__init__()

        # log hyperparameters, By calling save_hyperparameters we can ask lightning to save the values of anything in the __init__ for us to the checkpoint.
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_ = lambda_
        # n_sizes = self._get_conv_output(input_shape)
        self.n_channels = n_channels

        # coord = torch.randn(n_channels, n_channels, 4)
        self.W = nn.Parameter(torch.randn(n_channels, n_channels), requires_grad=True)

        # self.adj_fc = nn.Sequential(
        #     nn.Linear(4, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),  # nn.ReLU()
        # )
        # self.register_buffer("coord", coord)
        self.GC_layer = tgnn.ChebConv(
            in_channels=in_features,
            out_channels=out_features,
            K=K,
            normalization="sym",
        )

        self.GC_layer1 = tgnn.ChebConv(
            in_channels=out_features,
            out_channels=out_features,
            K=K,
            normalization="sym",
        )

        # self.GC_layer2 = tgnn.ChebConv(
        #     in_channels=out_features,
        #     out_channels=out_features,
        #     K=K,
        #     normalization="sym",
        # )

        # self.GC_layer = SpectralGNN(
        #     in_channels=in_features, out_channels=out_features, K=5, n_nodes=n_channels
        # )
        # self.GC_layer1 = SpectralGNN(
        #     in_channels=out_features, out_channels=out_features, K=5, n_nodes=n_channels
        # )
        # self.GC_layer2 = SpectralGNN(
        #     in_channels=out_features, out_channels=out_features, K=5, n_nodes=n_channels
        # )

        # self.GC_layer = GraphCN(in_features=in_features, out_features=out_features)
        # self.GC_layer1 = GraphCN(in_features=out_features, out_features=out_features)
        # self.GC_layer2 = GraphCN(in_features=out_features, out_features=out_features)

        self.conv11 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        # self.get_lambda_max = LaplacianLambdaMax(normalization="sym")
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_features * 2, n_classes),
        )

        self.accuracy = torchmetrics.Accuracy()

    # # returns the size of the output tensor going into Linear layer from the conv block.
    # def _get_conv_output(self, shape):
    #     batch_size = 1
    #     input = torch.autograd.Variable(torch.rand(batch_size, *shape))

    #     output_feat = self._forward_features(input)
    #     n_size = output_feat.data.view(batch_size, -1).size(1)
    #     return n_size

    # returns the feature tensor from the conv block
    # def _forward_features(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.pool1(F.relu(self.conv2(x)))
    #     x = F.relu(self.conv3(x))
    #     x = self.pool2(F.relu(self.conv4(x)))
    #     return x

    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)  # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x

    # will be used during inference
    def forward(self, x):

        # adj = self.relu(self.adj_fc(self.coord).squeeze())
        # # print(adj.shape)
        # # print(adj)

        # ## test ChebConv module in pytorch geometric
        # edge_index = (adj > 0).nonzero().t()
        # edge_weight = adj[edge_index[0], edge_index[1]]
        # print(self.W)
        edge_index = (self.W > 0).nonzero().t()
        edge_weight = self.W[edge_index[0], edge_index[1]]
        ## chebnet torch geometric

        b, n, _ = x.shape
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_weight.shape)
        # x = x.reshape(b * n, -1)
        x = self.GC_layer(x, edge_index=edge_index, edge_weight=edge_weight)
        # x = self.graph_max_pool(2)
        x = self.relu(x)

        x = self.GC_layer1(x, edge_index=edge_index, edge_weight=edge_weight)
        # x = self.graph_max_pool(2)
        x = self.relu(x)

        # x = self.GC_layer2(x, edge_index=edge_index, edge_weight=edge_weight)
        # # x = self.graph_max_pool(2)
        # x = self.relu(x)

        # x = x.reshape(b, n, -1)

        ## test GraphCN
        # x = self.GC_layer(x, adj)
        # x = self.GC_layer1(x, adj)
        # x = self.GC_layer1(x, adj)
        # x = x.unsqueeze(1)
        x = self.conv11(x)
        # print(torch.mean(x.squeeze(), dim=1).shape)
        # # x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = x.squeeze(1)
        # x = x[:, 0]

        out = self.fc(x)

        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # print(y.shape, logits.shape)
        # print(logits, y)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )

        return {
            "loss": loss,
            "acc": acc,
        }

    def training_epoch_end(self, outputs):

        self.W = nn.Parameter(torch.sgn(self.W) * torch.nn.functional.relu(torch.abs(self.W) -0.5* self.lambda_ * self.learning_rate, inplace = True), requires_grad = True)
        # print(new_W)
        # print(new_W.shape)
        # print(type(new_W))
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)

    #     # validation metrics
    #     preds = torch.argmax(logits, dim=1)
    #     acc = self.accuracy(preds, y)
    #     self.log("test_loss", loss, prog_bar=True)
    #     self.log("test_acc", acc, prog_bar=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


def log2txt(content, log_file):
    """
    this function log the content to results.txt
    :param content: string, the content to log
    """
    # log_file = "results.txt"
    file = open(log_file, "a")
    file.write(str(content) + "\n")
    file.close()


def get_best_weight(weight_dir=""):
    file_names = os.listdir(weight_dir)
    losses = []
    for name in file_names:
        loss = float(name.split("=")[3][:4])
        losses.append(loss)

    return file_names[np.argmin(losses)], int(
        file_names[np.argmin(losses)].split("=")[2].split("-")[0]
    )


def data_scaler(X_train, X_test, sc=None):
    _, c, f = X_train.shape
    X_train = X_train.reshape(-1, f)
    X_test = X_test.reshape(-1, f)
    if sc is None:
        sc = StandardScaler()
        sc.fit(X_train)

    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    return X_train.reshape(-1, c, f), X_test.reshape(-1, c, f)


def loto_train_segment(
    nth_subject=10,
    num_classes=2,
    inner_epochs=10,
    outer_epochs=10,
    label_type="V",
    folds=5,
):

    X, y_val, y_ar = extract_one_subject_features(
        sub_idx=nth_subject,
        num_classes=num_classes,
        save_dir="data",
        normalize_signal=True,
        segmenting=True,
        segment_duration=3,
    )
    if label_type == "V":
        y = y_val
    else:
        y = y_ar
    learning_rate = 1e-3
    batch_size = 64
    num_classes = 2
    dropout = 0.25
    kf_out = KFold(n_splits=40, shuffle=True, random_state=100)
    kf_out.get_n_splits(X)

    print(kf_out)
    predictions = []
    targets = []

    """======================================inner loop=============================================="""
    for train_index, test_index in kf_out.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        weight_dir = f"weights/sparseDCGNN_subject_{nth_subject:02d}/LOTO_segment_index_{test_index[0]}"
        kf = KFold(n_splits=folds, shuffle=True, random_state=29)
        kf.get_n_splits(X_train)
        sc = StandardScaler()

        for fold, (train_fold_index, val_index) in enumerate(kf.split(X_train)):
            X_train_fold, X_val = X_train[train_fold_index], X_train[val_index]
            y_train_fold, y_val = y_train[train_fold_index], y_train[val_index]
            sc.fit(X_train_fold.reshape(-1, X_train_fold.shape[-1]))
            X_train_fold, X_val = data_scaler(X_train_fold, X_val, sc=sc)

            train_fold_loader = get_dataloader(
                X_train_fold, y_train_fold, batch_size, shuffle=True, num_workers=0
            )
            val_loader = get_dataloader(
                X_val, y_val, batch_size, shuffle=False, num_workers=0
            )

            ##  initialize the model and trainer
            model = LitModel(
                in_features=X.shape[-1],
                out_features=32,
                n_channels=X.shape[-2],
                K=5,
                n_classes=num_classes,
                dropout_rate=dropout,
                learning_rate=learning_rate,
            )

            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)

            if not os.path.exists(
                f"lightning_logs/sparseDCGNN_segment_subjects_{nth_subject:02d}"
            ):
                os.makedirs(f"lightning_logs/sparseDCGNN_segment_subjects_{nth_subject:02d}")

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                save_weights_only=False,
                verbose=True,
                dirpath=weight_dir,
                filename=f"Fold={fold+1}" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
            )
            # logger = TensorBoardLogger(
            #     name=f"LOTO={val_index[0]}",
            #     save_dir=f"lightning_logs/TSception_subjects={nth_subject}",
            #     version=f"Fold={fold+1}",
            # )

            # logger = WandbLogger(
            #     project="sparseDCGNN",
            #     save_dir=f"lightning_logs/sparseDCGNN_subjects={nth_subject}",
            #     version=f"Fold={fold+1}",
            # )

            trainer = pl.Trainer(
                gpus=1,
                max_epochs=inner_epochs,
                # accumulate_grad_batches = 1,
                # auto_lr_find=True,
                callbacks=[checkpoint_callback],
                # logger=logger,
                # val_check_interval=0.25,
                check_val_every_n_epoch=1,
                # precision=32,
                # resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
                # default_save_path = './weights'
            )

            trainer.fit(model, train_fold_loader, val_loader)

        """===============================================outer loop===================================================="""
        X_train, X_test = data_scaler(X_train, X_test, sc=sc)

        train_loader = get_dataloader(
            X_train, y_train, batch_size=batch_size, num_workers=0
        )
        test_loader = get_dataloader(
            X_test, y_test, batch_size=batch_size, num_workers=0, shuffle=False
        )

        ## training code
        model = LitModel(
            in_features=X.shape[-1],
            out_features=32,
            n_channels=X.shape[-2],
            K=5,
            n_classes=num_classes,
            dropout_rate=dropout,
            learning_rate=learning_rate,
        )

        # Initialize wandb logger
        # wandb_logger = WandbLogger(project="sparseDCGNN-lightning", job_type="train")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_weights_only=False,
            verbose=True,
            dirpath=weight_dir,
            filename=f"final" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
        )
        # logger = WandbLogger(
        #     project="sparseDCGNN",
        #     save_dir=f"lightning_logs/sparseDCGNN_subjects={nth_subject}",
        #     version=f"final",
        # )

        weight_name, current_epoch = get_best_weight(weight_dir)

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=current_epoch + outer_epochs,
            # accumulate_grad_batches = 1,
            # auto_lr_find=True,
            callbacks=[checkpoint_callback],
            # logger=logger,
            # val_check_interval=0.25,
            check_val_every_n_epoch=1,
            # precision=16,
            resume_from_checkpoint=os.path.join(weight_dir, weight_name),
            # default_save_path = './weights'
        )

        trainer.fit(model, train_loader, test_loader)

        # Evaluate the model on the held-out test set ⚡⚡
        # trainer.test()

        # Close wandb run
        # wandb.finish()
        for x, labels in test_loader:
            print(x.shape)
            logits = model(x)
            logits = torch.argmax(logits, dim=1).squeeze().numpy()
            print(logits.shape)
            predictions.extend(list(logits))
            print(labels.squeeze().numpy().shape)
            targets.extend(list(labels.squeeze().numpy()))

    acc = accuracy_score(predictions, targets)
    f1 = f1_score(predictions, targets)
    print("accuracy: ", acc)
    print("f1 score:", f1)
    log_content = (
        f"subject {nth_subject}: \n"
        + f"predictions: {np.array(predictions).astype(np.uint8)} \n"
        + f"targets : {np.array(targets).astype(np.uint8)} \n"
        + f"accuracy score: {acc} \n"
        + f"f1 score: {f1} \n"
    )
    log2txt(content=log_content, log_file="results/loto_result.txt")


def loto_train(
    nth_subject=10,
    num_classes=2,
    inner_epochs=10,
    outer_epochs=10,
    label_type="V",
    folds=5,
):
    # 1. Start a W&B run
    X, y_val, y_ar = extract_one_subject_features(
        sub_idx=nth_subject,
        num_classes=num_classes,
        save_dir="data",
        normalize_signal=True,
    )
    if label_type == "V":
        y = y_val
    else:
        y = y_ar

    # wandb.init(project="CHEBNET_LEARNABLE_ADJM")
    # config = wandb.config
    learning_rate = 1e-3
    batch_size = 8
    num_classes = 2
    dropout = 0.25
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    print(loo)
    predictions = []
    targets = []

    """======================================inner loop=============================================="""
    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        weight_dir = (
            f"weights/sparseDCGNN_subject_{nth_subject:02d}/LOTO_index_{test_index[0]}"
        )
        kf = KFold(n_splits=folds, shuffle=True, random_state=29)
        kf.get_n_splits(X_train)
        # sc = StandardScaler()

        for fold, (train_fold_index, val_index) in enumerate(kf.split(X_train)):
            X_train_fold, X_val = X_train[train_fold_index], X_train[val_index]
            y_train_fold, y_val = y_train[train_fold_index], y_train[val_index]
            # sc.fit(X_train_fold.reshape(-1, X_train_fold.shape[-1]))
            # X_train_fold, X_val = data_scaler(X_train_fold, X_val, sc=sc)

            train_fold_loader = get_dataloader(
                X_train_fold, y_train_fold, batch_size, shuffle=True, num_workers=0
            )
            val_loader = get_dataloader(
                X_val, y_val, batch_size, shuffle=False, num_workers=0
            )

            ##  initialize the model and trainer
            model = LitModel(
                in_features=X.shape[-1],
                out_features=32,
                n_channels=X.shape[-2],
                K=5,
                n_classes=num_classes,
                dropout_rate=dropout,
                learning_rate=learning_rate,
            )

            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)

            if not os.path.exists(f"lightning_logs/sparseDCGNN_subjects_{nth_subject:02d}"):
                os.makedirs(f"lightning_logs/sparseDCGNN_subjects_{nth_subject:02d}")

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                save_weights_only=False,
                verbose=True,
                dirpath=weight_dir,
                filename=f"Fold={fold+1}" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
            )
            # logger = TensorBoardLogger(
            #     name=f"LOTO={val_index[0]}",
            #     save_dir=f"lightning_logs/TSception_subjects={nth_subject}",
            #     version=f"Fold={fold+1}",
            # )

            # logger = WandbLogger(
            #     project="sparseDCGNN",
            #     save_dir=f"lightning_logs/sparseDCGNN_subjects={nth_subject}",
            #     version=f"Fold={fold+1}",
            # )

            trainer = pl.Trainer(
                gpus=1,
                max_epochs=inner_epochs,
                # accumulate_grad_batches = 1,
                # auto_lr_find=True,
                callbacks=[checkpoint_callback],
                # logger=logger,
                # val_check_interval=0.25,
                check_val_every_n_epoch=1,
                # precision=32,
                # resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
                # default_save_path = './weights'
            )

            trainer.fit(model, train_fold_loader, val_loader)

        """===============================================outer loop===================================================="""
        # X_train, X_test = data_scaler(X_train, X_test, sc=sc)

        train_loader = get_dataloader(
            X_train, y_train, batch_size=batch_size, num_workers=0
        )
        # val_loader = get_dataloader(
        #     X_test, y_test, batch_size=1, num_workers=0, shuffle=False
        # )

        ## training code
        model = LitModel(
            in_features=X.shape[-1],
            out_features=32,
            n_channels=X.shape[-2],
            K=5,
            n_classes=num_classes,
            dropout_rate=dropout,
            learning_rate=learning_rate,
        )

        # Initialize wandb logger
        # wandb_logger = WandbLogger(project="sparseDCGNN-lightning", job_type="train")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_weights_only=False,
            verbose=True,
            dirpath=weight_dir,
            filename=f"final" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
        )
        # logger = WandbLogger(
        #     project="sparseDCGNN",
        #     save_dir=f"lightning_logs/sparseDCGNN_subjects={nth_subject}",
        #     version=f"final",
        # )

        weight_name, current_epoch = get_best_weight(weight_dir)

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=current_epoch + outer_epochs,
            # accumulate_grad_batches = 1,
            # auto_lr_find=True,
            callbacks=[checkpoint_callback],
            # logger=logger,
            # val_check_interval=0.25,
            check_val_every_n_epoch=1,
            # precision=16,
            resume_from_checkpoint=os.path.join(weight_dir, weight_name),
            # default_save_path = './weights'
        )

        trainer.fit(model, train_loader)  # , val_loader)

        # Evaluate the model on the held-out test set ⚡⚡
        # trainer.test()

        # Close wandb run
        # wandb.finish()

        logits = model(torch.Tensor(X_test))
        predictions.append(torch.argmax(logits, dim=1).squeeze().numpy())
        targets.append(y_test.squeeze())

    acc = accuracy_score(predictions, targets)
    f1 = f1_score(predictions, targets)
    print("accuracy: ", acc)
    print("f1 score:", f1)
    log_content = (
        f"subject {nth_subject}: \n"
        + f"predictions: {np.array(predictions).astype(np.uint8)} \n"
        + f"targets : {np.array(targets).astype(np.uint8)} \n"
        + f"accuracy score: {acc} \n"
        + f"f1 score: {f1} \n"
    )
    log2txt(content=log_content, log_file="results/loto_result.txt")

def train_DEAP(batch_size = 16, dropout = 0.2, learning_rate = 1e-3, epochs = 20):
    # if DE_DCGNN_full.npy then the number of bands are 4
    X, y_valence, y_arousal, y_dominance = np.load("data/DEAP_DE_DCGNN_full.npy"), np.load("data/DEAP_valence_DE_DCGNN.npy"), np.load("data/DEAP_arousal_DE_DCGNN.npy"), np.load("data/DEAP_dominance_DE_DCGNN.npy")
    print(X.shape, y_valence.shape)
    num_classes = 2
    # current date and time
    curDT = datetime.now()
    date_time = curDT.strftime("%m/%d/%Y, %H:%M:%S")
    log2txt(content=f"\n\n4 bands\nleave-one-trial-out\ndate and time: {date_time}\n", log_file="results/loto_result.txt")
    del date_time
    del curDT
    # log_content = f"\n\n4 bands\nleave-one-trial-out\ndate and time: {date_time}\n"
    label_types = ["valence", "arousal", "dominance"]
    for i, y in enumerate([y_valence]):#, y_arousal, y_dominance]):
        # log_content += f"\nlabel type: {label_types[i]}\n"
        log2txt(content=f"\nlabel type: {label_types[i]}\n", log_file="results/loto_result.txt")

        for sub in range(X.shape[0]):
            # log_content += f"subject number {sub+1}:\n"
            log2txt(content=f"subject number {sub+1}:\n", log_file="results/loto_result.txt")

            X_sub = X[sub] # shape = 18x14x3
            y_sub = y[sub] # shape = 18x14x1
            y_sub = np.where(y_sub>5, 1, 0) # convert to high/low
            kf = KFold(n_splits=40, shuffle=False) # leave each trial out
            # kf.get_n_splits(X_sub)
            accs = []
            f1s = []

            for fold, (train_fold_index, val_index) in enumerate(kf.split(X_sub)):
                X_train, X_val = X_sub[train_fold_index], X_sub[val_index]
                y_train, y_val = y_sub[train_fold_index], y_sub[val_index]
                X_train = X_train.reshape(-1, X.shape[-2], X.shape[-1])
                X_val = X_val.reshape(-1, X.shape[-2], X.shape[-1])
                y_train = y_train.reshape(-1, )
                y_val = y_val.reshape(-1, )

                print(X_val.shape, y_val.shape)

                # print(y_sub)
                # X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, stratify = y_sub, test_size = 0.2, shuffle = True, random_state = 29)
                train_loader = get_dataloader(
                        X_train, y_train, batch_size, shuffle=True, num_workers=0
                    )
                val_loader = get_dataloader(
                    X_val, y_val, batch_size, shuffle=False, num_workers=0
                )

                weight_dir = f"weights/4bands_sparseDCGNN_subject_{sub+1:02d}/LOTO_index_{fold:02d}"

                ##  initialize the model and trainer
                model = LitModel(
                    in_features=X_sub.shape[-1],
                    out_features=32,
                    n_channels=X_sub.shape[-2],
                    K=5,
                    n_classes=num_classes,
                    dropout_rate=dropout,
                    learning_rate=learning_rate,
                )

                if not os.path.exists(weight_dir):
                    os.makedirs(weight_dir)

                log_dir = f"lightning_logs/4bands_sparseDCGNN_subject_{sub+1:02d}"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
                checkpoint_callback = ModelCheckpoint(
                    monitor="val_loss",
                    save_weights_only=False,
                    verbose=True,
                    dirpath=weight_dir,
                    filename=f"4bands_sparseDCGNN-"+"-{epoch:02d}-{val_loss:.2f}.ckpt",
                )
                # logger = TensorBoardLogger(
                #     name=f"4bands_LOTO={fold+1}",
                #     save_dir=log_dir,
                #     # version=f"Fold={fold+1}",
                # )

                # logger = WandbLogger(
                #     project="sparseDCGNN",
                #     save_dir=f"lightning_logs/sparseDCGNN_subjects={nth_subject}",
                #     version=f"Fold={fold+1}",
                # )

                trainer = pl.Trainer(
                    gpus=1,
                    max_epochs=epochs,
                    # accumulate_grad_batches = 1,
                    # auto_lr_find=True,
                    callbacks=[checkpoint_callback],#, early_stop_callback],
                    # logger=logger,
                    # val_check_interval=0.25,
                    check_val_every_n_epoch=1,
                    # precision=32,
                    # resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
                    # default_save_path = './weights'
                )

                trainer.fit(model, train_loader, val_loader)
                # print("loading best model from: ",checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
                # print("score of the best model: ", checkpoint_callback.best_model_score) # and prints it score
                model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
                y_true, y_pred, y_pred_prob = pytorch_predict(model, val_loader, "cuda")
                acc = accuracy_score(y_pred, y_true)
                f1 =  f1_score(y_pred, y_true)
                accs.append(acc)
                f1s.append(f1)
                print(acc, f1)

            # log_content += f"{np.mean(accs)}/{np.std(accs)} , {np.mean(f1s)}/{np.std(f1s)} \n"
            log2txt(content=f"{np.mean(accs)}/{np.std(accs)} , {np.mean(f1s)}/{np.std(f1s)} \n", log_file="results/loto_result.txt")


    # log2txt(content=log_content, log_file="results/loto_result.txt")

def train_DEAP_LOSO(batch_size = 16, dropout = 0.2, learning_rate = 1e-3, epochs = 20):
    X, y_valence, y_arousal, y_dominance = np.load("data/DEAP_DE_DCGNN_full.npy"), np.load("data/DEAP_valence_DE_DCGNN.npy"), np.load("data/DEAP_arousal_DE_DCGNN.npy"), np.load("data/DEAP_dominance_DE_DCGNN.npy")
    print(X.shape, y_valence.shape)
    num_classes = 2
    # current date and time
    curDT = datetime.now()
    date_time = curDT.strftime("%m/%d/%Y, %H:%M:%S")

    # log_content = f"\n\n4 bands\nleave-one-subject-out\ndate and time: {date_time}\n"
    log2txt(content=f"\n\n4 bands\nleave-one-subject-out\ndate and time: {date_time}\n", log_file="results/loto_result.txt")
    del date_time
    del curDT
    label_types = ["valence", "arousal", "dominance"]
    for i, y in enumerate([y_valence, y_arousal, y_dominance]):
        # log_content += f"\nlabel type: {label_types[i]}\n"
        log2txt(content=f"\nlabel type: {label_types[i]}\n", log_file="results/loto_result.txt")


        y = np.where(y>5, 1, 0) # convert to high/low
        kf = KFold(n_splits=32, shuffle=False) # leave each trial out
        # kf.get_n_splits(X_sub)
        accs = []
        f1s = []

        for fold, (train_fold_index, val_index) in enumerate(kf.split(X)):
            # log_content += f"subject number {fold+1}:\n"
            log2txt(content=f"subject number {fold+1}:\n", log_file="results/loto_result.txt")

            X_train, X_val = X[train_fold_index], X[val_index]
            y_train, y_val = y[train_fold_index], y[val_index]
            X_train = X_train.reshape(-1, X.shape[-2], X.shape[-1])
            X_val = X_val.reshape(-1, X.shape[-2], X.shape[-1])
            y_train = y_train.reshape(-1, )
            y_val = y_val.reshape(-1, )

            print(X_val.shape, y_val.shape)

            # print(y_sub)
            # X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, stratify = y_sub, test_size = 0.2, shuffle = True, random_state = 29)
            train_loader = get_dataloader(
                    X_train, y_train, batch_size, shuffle=True, num_workers=0
                )
            val_loader = get_dataloader(
                X_val, y_val, batch_size, shuffle=False, num_workers=0
            )

            weight_dir = f"weights/4bands_LOSO_sparseDCGNN_subject_{fold+1:02d}"

            ##  initialize the model and trainer
            model = LitModel(
                in_features=X.shape[-1],
                out_features=32,
                n_channels=X.shape[-2],
                K=5,
                n_classes=num_classes,
                dropout_rate=dropout,
                learning_rate=learning_rate,
            )

            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)

            log_dir = f"lightning_logs/4bands_LOSO_sparseDCGNN_subject_{fold+1:02d}"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=10, verbose=False, mode="max")
            checkpoint_callback = ModelCheckpoint(
                monitor="val_acc",
                mode = "max",
                save_weights_only=False,
                verbose=True,
                dirpath=weight_dir,
                filename=f"4bands_LOSO_sparseDCGNN-"+"-{epoch:02d}-{val_accuracy:.2f}.ckpt",
            )
            # logger = TensorBoardLogger(
            #     name=f"4bands_LOSO={fold+1}",
            #     save_dir=log_dir,
            #     # version=f"Fold={fold+1}",
            # )

            # logger = WandbLogger(
            #     project="sparseDCGNN",
            #     save_dir=f"lightning_logs/sparseDCGNN_subjects={nth_subject}",
            #     version=f"Fold={fold+1}",
            # )

            trainer = pl.Trainer(
                gpus=1,
                max_epochs=epochs,
                # accumulate_grad_batches = 1,
                # auto_lr_find=True,
                callbacks=[checkpoint_callback],#, early_stop_callback],
                # logger=logger,
                # val_check_interval=0.25,
                check_val_every_n_epoch=1,
                # precision=32,
                # resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
                # default_save_path = './weights'
            )

            trainer.fit(model, train_loader, val_loader)
            # print("loading best model from: ",checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
            # print("score of the best model: ", checkpoint_callback.best_model_score) # and prints it score
            model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
            y_true, y_pred, y_pred_prob = pytorch_predict(model, val_loader, "cuda")
            acc = accuracy_score(y_pred, y_true)
            f1 =  f1_score(y_pred, y_true)
            accs.append(acc)
            f1s.append(f1)
            print(acc, f1)

        # log_content += f"{np.mean(accs)}/{np.std(accs)} , {np.mean(f1s)}/{np.std(f1s)} \n"
        log2txt(content=f"{np.mean(accs)}/{np.std(accs)} , {np.mean(f1s)}/{np.std(f1s)} \n", log_file="results/loto_result.txt")

    # log2txt(content=log_content, log_file="results/loso_result.txt")


def pytorch_predict(model, test_loader, device):
    '''
    Make prediction from a pytorch model
    '''
    # set model to evaluate model
    model.to(device)
    model.eval()

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)

            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()

    return y_true, y_pred, y_pred_prob

if __name__ == "__main__":
    # 1. Start a W&B run
    # wandb.init(project="gpt4")
    # config = wandb.config
    # config.learning_rate = 1e-3
    # config.num_classes = 2

    # preprocessing data
    ## 1 subject data
    # X, y_val, y_ar = extract_one_subject_features(
    #     sub_idx=10, num_classes=2, save_dir="data"
    # )

    # train_loader = get_dataloader(X, y_val, batch_size=32, num_workers=0)
    # val_loader = None
    ## all data
    # X, y_val, y_ar = extract_all_features()

    # X = np.load("data/all_de_features.npy")
    # y_val = np.load("data/all_val2_labels.npy")
    # y_ar = np.load("data/all_ar2_labels.npy")
    # X_train, y_train = X[: 40 * 24], y_val[: 40 * 24]
    # X_test, y_test = X[40 * 24 :], y_val[40 * 24 :]
    # sc = StandardScaler()
    # X_train, X_test = data_scaler(X_train, X_test, sc = None)
    # # print(X.shape, y_val.shape, y_ar.shape)
    # train_loader = get_dataloader(
    #     X[: 40 * 24], y_val[: 40 * 24], batch_size=32, num_workers=0
    # )
    # val_loader = get_dataloader(
    #     X[40 * 24 :], y_val[40 * 24 :], batch_size=32, num_workers=0, shuffle=False
    # )

    # ## training code
    # model = LitModel(
    #     in_features=X.shape[-1],
    #     out_features=32,
    #     n_channels=X.shape[-2],
    #     K=5,
    #     n_classes=config.num_classes,
    #     dropout_rate=0.0,
    #     learning_rate=config.learning_rate,
    # )

    # # Initialize wandb logger
    # wandb_logger = WandbLogger(project="wandb-lightning", job_type="train")

    # # Initialize a trainer
    # trainer = pl.Trainer(
    #     max_epochs=2000,
    #     progress_bar_refresh_rate=20,
    #     gpus=1,
    #     logger=wandb_logger,
    #     # callbacks=[early_stop_callback, ImagePredictionLogger(val_samples)],
    #     # checkpoint_callback=checkpoint_callback,
    # )

    # # Train the model ⚡🚅⚡
    # trainer.fit(model, train_loader, val_loader)

    # # Evaluate the model on the held-out test set ⚡⚡
    # # trainer.test()

    # # Close wandb run
    # wandb.finish()
    # """===================================train loto no segmenting======================================"""
    # t_start = time.time()

    # for idx in range(1, 33):
    #     loto_train(inner_epochs=150, outer_epochs=10, nth_subject=idx, folds=3)
    # log2txt(
    #     content="Execution time: {:.2f} minutes".format((time.time() - t_start) // 60),
    #     log_file="results/loto_result.txt",
    # )
    # """===================================end train loto no segmenting=================================="""

    """===================================train loto segmenting========================================="""
    # t_start = time.time()

    # for idx in range(1, 2):
    #     loto_train_segment(inner_epochs=50, outer_epochs=10, nth_subject=idx, folds=5)
    # log2txt(
    #     content="Execution time: {:.2f} minutes".format((time.time() - t_start) // 60),
    #     log_file="results/loto_result.txt",
    # )
    """===================================end train loto segmenting====================================="""

    """train subject dependent segmenting shuffle sparseDCGNN"""
    t_start = time.time()

    train_DEAP(epochs=30, batch_size =256)

    # train_DEAP_LOSO(epochs = 50, batch_size = 256)
    print("Execution time: {:.2f} minutes".format((time.time() - t_start) // 60))
