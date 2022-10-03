from atexit import register
import torch.nn as nn
import torch

import numpy as np
from utils import *
from decimal import MAX_EMAX
from pkgutil import get_loader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from DCGNN import DCGNN

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from imblearn.over_sampling import RandomOverSampler
from utils import get_one_subject, get_dataloader

from einops import rearrange
import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import argparse
import datetime

from utils import *


DEVICE = torch.device("cuda:0")
NUM_GPUS = 1
EPOCHS = 25
CUDA = torch.cuda.is_available()
LOG_FILE = "results.txt"


def log2txt(content):
    """
    this function log the content to results.txt
    :param content: string, the content to log
    """
    # log_file = "results.txt"
    file = open(LOG_FILE, "a")
    file.write(str(content) + "\n")
    file.close()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


def get_best_weight(weight_dir=""):
    file_names = os.listdir(weight_dir)
    losses = []
    for name in file_names:
        loss = float(name.split("=")[3][:4])
        losses.append(loss)

    return file_names[np.argmin(losses)], int(
        file_names[np.argmin(losses)].split("=")[2][:2]
    )


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_func, regularized=1):
    model.train()
    tl = Averager()
    pred_train = []
    label_train = []
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        out = model(x_batch)
        # print(out.shape, y_batch.shape)
        loss = loss_func(out, y_batch)
        # print(out, y_batch)
        # print(loss)
        _, pred = torch.max(out, 1)

        ## l2 regularization
        if regularized:
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(regularized)
                else:
                    l2_reg = l2_reg + W.norm(regularized)
            loss += l2_reg

        tl.add(loss)
        pred_train.extend(pred.data.tolist())
        label_train.extend(y_batch.data.tolist())
        # torch.autograd.set_detect_anomaly(True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    # print(model.W_star)
    return tl.item(), pred_train, label_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, train_loader, val_loader, subject, fold):
    seed_all(args.random_seed)
    save_name = "_sub" + str(subject) + "_trial" + str(fold)
    model_name_reproduce = ""
    set_up(args)
    params = list(model.parameters())  # + list(model.W_star)
    optimizer = torch.optim.SGD(
        params,
        lr=args.learning_rate1,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.9
    )

    loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, "{}.pth".format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, "{}.pth".format(name)))

    # initiate log dictionary
    trlog = {}
    trlog["args"] = vars(args)
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["train_acc"] = []
    trlog["val_acc"] = []
    trlog["max_acc"] = 0.0

    # timer = Timer()

    for epoch in range(args.max_epoch):

        loss_train, pred_train, label_train = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            regularized=args.regularized,
        )
        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=label_train)
        print(
            "epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}".format(
                epoch, loss_train, acc_train, f1_train
            )
        )
        loss_val, pred_val, label_val = predict(val_loader, model, loss_fn)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=label_val)
        print(
            "epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}".format(
                epoch, loss_val, acc_val, f1_val
            )
        )
        if acc_val > trlog["max_acc"]:
            trlog["max_acc"] = acc_val
            save_model("max-acc")

            if args.save_model:
                # save model here for reproduce
                model_name_reproduce = (
                    "sub"
                    + str(subject)
                    + "_fold"
                    + str(fold)
                    + str(round(trlog["max_acc"], 2))
                    + ".pth"
                )
                data_type = "model_{}_{}".format(args.dataset, args.label_type)
                save_path = osp.join(args.save_path, data_type)
                ensure_path(save_path)
                model_name_reproduce = osp.join(save_path, model_name_reproduce)
                torch.save(model.state_dict(), model_name_reproduce)

        trlog["train_loss"].append(loss_train)
        trlog["train_acc"].append(acc_train)
        trlog["val_loss"].append(loss_val)
        trlog["val_acc"].append(acc_val)

        # print(timer.measure())
        # print(
        #     "ETA:{}/{} SUB:{} FOLD:{}".format(
        #         timer.measure(), timer.measure(epoch / args.max_epoch), subject, fold
        #     )
        # )
    save_name_ = "trlog" + save_name
    ensure_path(osp.join(args.save_path, "log_train"))
    torch.save(trlog, osp.join(args.save_path, "log_train", save_name_))

    return trlog["max_acc"], model_name_reproduce


def train_main(args):
    # Train and evaluate the model subject by subject
    tta = []  # total test accuracy
    tva = []  # total validation accuracy
    ttf = []  # total test f1

    BATCH_SIZE = 8
    # get data of one subject
    X_origin = np.load("data/all_de_features.npy")
    if args.label_type == "V":
        y_origin = np.load("data/all_val2_labels.npy")
    else:
        y_origin = np.load("data/all_ar2_labels.npy")

    for SUBJECT_NUMBER in range(1, args.subjects + 1):
        X = X_origin[40 * (SUBJECT_NUMBER - 1) : 40 * SUBJECT_NUMBER]
        y = y_origin[40 * (SUBJECT_NUMBER - 1) : 40 * SUBJECT_NUMBER].squeeze()
        ## implement leave one trial out
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        predictions = []
        targets = []
        for fold, (train_index, test_index) in enumerate(loo.split(X)):
            print("TRAIN:", train_index, "VAL:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("X_train shape: ", X_train.shape)
            print("y_train shape: ", y_train.shape)
            # # reshape before normalize -> (n 1 c s)
            # X_train = rearrange(X_train, "n c s -> n 1 c s", c=32)
            # X_test = rearrange(X_test, "n c s -> n 1 c s", c=32)
            # X_train, X_test = normalize(X_train, X_test)

            # ## duplicate minority label
            # if args.classes_balanced:
            #     X_train = rearrange(X_train, "n 1 c s -> n (c s)")
            #     oversample = RandomOverSampler(sampling_strategy="minority")
            #     X_train, y_train = oversample.fit_resample(X_train, y_train)
            #     X_train = rearrange(X_train, "n (c s) -> n 1 c s", c=32)

            ## implement k-fold
            # weight_dir = f"weights/LGGNet_subject_{SUBJECT_NUMBER:02d}/LOTO_index_{test_index[0]}"
            kf = KFold(n_splits=args.inner_folds)
            kf.get_n_splits(X_train)
            max_fold_acc = 0
            best_model_name = ""
            fold_acc = []

            for inner_fold, (train_fold_index, val_index) in enumerate(
                kf.split(X_train)
            ):
                print(
                    "Subject: {} - Fold: {} - Inner_Fold: {}".format(
                        SUBJECT_NUMBER, fold, inner_fold
                    )
                )
                X_train_fold, X_val = X_train[train_fold_index], X_train[val_index]
                y_train_fold, y_val = y_train[train_fold_index], y_train[val_index]

                train_loader = get_dataloader(
                    X_train_fold, y_train_fold, args.batch_size, shuffle=True
                )
                val_loader = get_dataloader(
                    X_val, y_val, args.batch_size, shuffle=False
                )

                ##  initialize the model
                model = DCGNN(
                    in_features=4,
                    out_features=16,
                    n_channels=32,
                    K=5,
                    n_classes=args.num_class,
                    dropout_rate=args.dropout,
                )
                if CUDA:
                    model = model.cuda()

                acc_val, model_name = train(
                    args, model, train_loader, val_loader, SUBJECT_NUMBER, inner_fold
                )

                fold_acc.append(acc_val)

                if max_fold_acc < round(acc_val, 2):
                    max_fold_acc = round(acc_val, 2)
                    best_model_name = model_name

            ## outer loop : fine-tune the best model for 5 epochs.
            # load the model with name given by best accuracy
            model = DCGNN(
                in_features=4,
                out_features=16,
                n_channels=32,
                K=5,
                n_classes=args.num_class,
                dropout_rate=args.dropout,
            )
            if CUDA:
                model = model.cuda()
            model.load_state_dict(torch.load(best_model_name))
            # initiate optimizer and loss function
            params = list(model.parameters())  # + list(model.W_star)

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate2,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=np.arange(10, 50, 10), gamma=0.9
            )
            loss_fn = nn.CrossEntropyLoss()

            # create data loaders
            train_loader = get_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
            test_loader = get_dataloader(X_test, y_test, BATCH_SIZE, shuffle=False)
            # train the model 5 more epochs on the whole training data.
            print("====================================================")
            print("Start fine-tuning best model for 5 epochs.")
            for epoch in range(args.inner_folds):
                loss_train, pred_train, label_train = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    regularized=args.regularized,
                )
                acc_train, f1_train, _ = get_metrics(
                    y_pred=pred_train, y_true=label_train
                )
                print(
                    "epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}".format(
                        epoch, loss_train, acc_train, f1_train
                    )
                )

            # save final model
            torch.save(
                model.state_dict(),
                osp.join(best_model_name[:-6] + ".pth"),
            )
            loss_test, pred_test, label_test = predict(
                data_loader=test_loader, net=model, loss_fn=loss_fn
            )

            predictions.append(pred_test)
            targets.append(label_test)

        # print(predictions, "\n", targets)
        acc_test, f1_test, _ = get_metrics(y_pred=predictions, y_true=targets)

        tva.append(np.mean(fold_acc))

        tta.append(acc_test)
        ttf.append(f1_test)

        result = "{},{}".format(tta[-1], ttf[-1])
        log2txt(result)
    # prepare final report
    mACC = np.mean(tta)
    mF1 = np.mean(ttf)
    std = np.std(tta)
    mACC_val = np.mean(tva)
    std_val = np.std(tva)

    print("Final: test mean ACC:{} std:{}".format(mACC, std))
    print("Final: val mean ACC:{} std:{}".format(mACC_val, std_val))
    results = "test mAcc={} mF1={} val mAcc={}".format(mACC, mF1, mACC_val)
    log2txt(results)


def train_loso(args):
    # Train and evaluate the model subject by subject
    tta = []  # total test accuracy
    tva = []  # total validation accuracy
    ttf = []  # total test f1

    BATCH_SIZE = 8
    # get data of one subject
    X = np.load("data/all_de_features.npy")
    if args.label_type == "V":
        y = np.load("data/all_val2_labels.npy")
        y = y.squeeze()
    else:
        y = np.load("data/all_ar2_labels.npy")
        y = y.squeeze()
    loso = KFold()
    loso.get_n_splits(X)
    for SUBJECT_NUMBER, (train_index, test_index) in enumerate(loso.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        # # reshape before normalize -> (n 1 c s)
        # X_train = rearrange(X_train, "n c s -> n 1 c s", c=32)
        # X_test = rearrange(X_test, "n c s -> n 1 c s", c=32)
        # X_train, X_test = normalize(X_train, X_test)

        # ## duplicate minority label
        # if args.classes_balanced:
        #     X_train = rearrange(X_train, "n 1 c s -> n (c s)")
        #     oversample = RandomOverSampler(sampling_strategy="minority")
        #     X_train, y_train = oversample.fit_resample(X_train, y_train)
        #     X_train = rearrange(X_train, "n (c s) -> n 1 c s", c=32)

        ## implement k-fold
        # weight_dir = f"weights/LGGNet_subject_{SUBJECT_NUMBER:02d}/LOTO_index_{test_index[0]}"
        kf = KFold(n_splits=args.inner_folds)
        kf.get_n_splits(X_train)
        max_fold_acc = 0
        best_model_name = ""
        fold_acc = []

        for inner_fold, (train_fold_index, val_index) in enumerate(kf.split(X_train)):
            print("Subject: {} - Fold: {}".format(SUBJECT_NUMBER + 1, inner_fold))
            X_train_fold, X_val = X_train[train_fold_index], X_train[val_index]
            y_train_fold, y_val = y_train[train_fold_index], y_train[val_index]
            print("X_train fold shape: ", X_train_fold.shape)
            print("y_train_fold shape: ", y_train_fold.shape)
            train_loader = get_dataloader(
                X_train_fold, y_train_fold, args.batch_size, shuffle=True
            )
            val_loader = get_dataloader(X_val, y_val, args.batch_size, shuffle=False)

            ##  initialize the model
            model = DCGNN(
                in_features=4,
                out_features=16,
                n_channels=32,
                K=5,
                n_classes=args.num_class,
                dropout_rate=args.dropout,
            )
            if CUDA:
                model = model.cuda()

            acc_val, model_name = train(
                args, model, train_loader, val_loader, SUBJECT_NUMBER, inner_fold
            )

            fold_acc.append(acc_val)

            if max_fold_acc < round(acc_val, 2):
                max_fold_acc = round(acc_val, 2)
                best_model_name = model_name

        ## outer loop : fine-tune the best model for 5 epochs.
        # load the model with name given by best accuracy
        model = DCGNN(
            in_features=4,
            out_features=16,
            n_channels=32,
            K=5,
            n_classes=args.num_class,
            dropout_rate=args.dropout,
        )
        if CUDA:
            model = model.cuda()
        model.load_state_dict(torch.load(best_model_name))
        # # initiate optimizer and loss function
        # params = list(model.parameters())  # + list(model.W_star)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate2,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=np.arange(10, 50, 10), gamma=0.9
        )
        loss_fn = nn.CrossEntropyLoss()

        # create data loaders
        train_loader = get_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
        test_loader = get_dataloader(X_test, y_test, BATCH_SIZE, shuffle=False)
        # train the model 5 more epochs on the whole training data.
        print("====================================================")
        print("Start fine-tuning best model for 5 epochs.")
        for epoch in range(5):
            loss_train, pred_train, label_train = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_fn,
                regularized=args.regularized,
            )
            acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=label_train)
            print(
                "epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}".format(
                    epoch, loss_train, acc_train, f1_train
                )
            )

        # save final model
        torch.save(
            model.state_dict(),
            osp.join(best_model_name[:-6] + ".pth"),
        )
        loss_test, pred_test, label_test = predict(
            data_loader=test_loader, net=model, loss_fn=loss_fn
        )

        # print(predictions, "\n", targets)
        acc_test, f1_test, _ = get_metrics(y_pred=pred_test, y_true=label_test)

        tva.append(np.mean(fold_acc))

        tta.append(acc_test)
        ttf.append(f1_test)

        result = "{},{}".format(tta[-1], ttf[-1])
        log2txt(result)
    # prepare final report
    mACC = np.mean(tta)
    mF1 = np.mean(ttf)
    std = np.std(tta)
    mACC_val = np.mean(tva)
    std_val = np.std(tva)

    print("Final: test mean ACC:{} std:{}".format(mACC, std))
    print("Final: val mean ACC:{} std:{}".format(mACC_val, std_val))
    results = "test mAcc={} mF1={} val mAcc={}".format(mACC, mF1, mACC_val)
    log2txt(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument("--dataset", type=str, default="DEAP")
    # parser.add_argument("--data-path", type=str, default="/home/dingyi/data/deap/")
    parser.add_argument("--subjects", type=int, default=32)
    parser.add_argument("--num-class", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--label-type", type=str, default="A", choices=["A", "V"])
    # parser.add_argument(
    #     "--segment", type=int, default=4, help="segment length in seconds"
    # )
    # parser.add_argument(
    #     "--trial-duration", type=int, default=60, help="trial duration in seconds"
    # )
    # parser.add_argument("--overlap", type=float, default=0)
    # parser.add_argument("--sampling-rate", type=int, default=128)
    parser.add_argument("--input-shape", type=tuple, default=(1, 32, 7680))
    # parser.add_argument("--data-format", type=str, default="raw")
    ######## Training Process ########
    parser.add_argument("--random-seed", type=int, default=2021)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate1", type=float, default=1e-3)
    parser.add_argument("--learning-rate2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--regularized", type=int, default=0)

    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--classes-balanced", type=int, default=1)
    parser.add_argument("--save-path", default="./save/")
    parser.add_argument("--load-path", default="./save/max-acc.pth")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--save-model", type=bool, default=True)
    parser.add_argument("--train-type", type=str, default="loto")
    ######## Model Parameters ########
    parser.add_argument("--model", type=str, default="DCGNN")
    # parser.add_argument(
    #     "--graph-type",
    #     type=str,
    #     default="TS",
    #     choices=["TS", "O"],
    #     help="TS for the channel order of TSception, O for the original channel order",
    # )
    parser.add_argument("--hidden", type=int, default=32)

    ######## Reproduce the result using the saved model ######
    # parser.add_argument("--reproduce", type=bool, default=False)
    args = parser.parse_args()

    file = open(LOG_FILE, "a")
    file.write(
        "\n"
        + str(datetime.datetime.now())
        + "\nTrain:Parameter setting for "
        + str(args.model)
        + " on "
        + str(args.dataset)
        + "\n1)number_class:"
        + str(args.num_class)
        + "\n2)random_seed:"
        + str(args.random_seed)
        + "\n3)learning_rate1:"
        + str(args.learning_rate1)
        + "\n3)learning_rate2:"
        + str(args.learning_rate2)
        + "\n4)weight_decay:"
        + str(args.weight_decay)
        + "\n5)num_epochs:"
        + str(args.max_epoch)
        + "\n6)batch_size:"
        + str(args.batch_size)
        + "\n7)dropout:"
        + str(args.dropout)
        + "\n8)hidden_node:"
        + str(args.hidden)
        + "\n9)input_shape:"
        + str(args.input_shape)
        + "\n10)class:"
        + str(args.label_type)
        + "\n11)inner_folds:"
        + str(args.inner_folds)
        + "\n12)classes_balanced:"
        + str(bool(args.classes_balanced))
        + "\n13)regularized:"
        + str(args.regularized)
        + "\n14)train_type:"
        + str(args.train_type)
        + "\n"
    )
    file.close()

    if args.train_type == "loso":
        train_loso(args)
    else:
        train_main(args)
