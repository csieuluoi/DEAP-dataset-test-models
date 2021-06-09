from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

# from custom_transform import CustomTransform
from custom_dataset import DEAP_Fnet_dataset

import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
# from AdasOptimizer.adasopt_pytorch import Adas

import inspect
import os
import random
import numpy as np
from tqdm import tqdm
from Fnet import FNet
from MLPmixer import MLPMixer
from RNN import CRNN
from utils import load_DEAP
from custom_transforms import CustomTransform
import pandas as pd

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', action='store', type=float, default = 1e-3, required = False)
parser.add_argument('--batch_size', action='store', type=int, default = 64, required = False)
parser.add_argument('--num_epochs', action='store', type=int, default = 5, required = False)

args = parser.parse_args()


BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
NUM_WORKERS = 1
# DATA_FILE_NAME = "EEG_frontal_segmented_6_2.npy"
# LABEL_FILE_NAME = "label_segmented_6_2.npy"
# ROOT_DIR = ""
# LABEL_TYPE = 0
TRANSFORMS = CustomTransform(normalized = 'Standard')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            preds.append(predictions)
        print(
            f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}"
        )
    print(preds)

def save_checkpoint(state, epoch, model_name = ""):
    if not os.path.exists(f"weights/{model_name}"):
        os.mkdir(f"weights/{model_name}")
    filename =f"weights/{model_name}/{model_name}_{epoch}.pth.tar"
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer
def train_fn(model, criterion, optimizer, epoch, writer, train_loader, test_loader, model_name):
    for epoch in range(NUM_EPOCHS):

        losses = []
        num_correct = 0
        num_samples = 0
        model.train()
        print(f"***Epoch {epoch}/{NUM_EPOCHS}***")
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for batch_idx, (data, targets) in loop:
            # print(targets)
            data = data.to(device = device)
            targets = targets.to(device = device)

            scores = model(data)

            loss = criterion(scores, targets)

            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            # print(scores)
            # calculate accuracy
            _, predictions = scores.max(1)
            num_correct += (predictions==targets).sum().item()
            num_samples += predictions.size(0)


            # update progess bar
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss = loss.item(), acc = float(num_correct)/float(num_samples)*100)
        print(f"train loss at epoch {epoch} is {sum(losses)/len(losses):.5f},  train accuracy is {float(num_correct)/float(num_samples)*100}")

                # printing test loss, accuracy
        model.eval()
        n_correct = 0
        n_samples = 0
        t_losses = []
        with torch.no_grad():
            loop_val = tqdm(enumerate(test_loader), total = len(test_loader), leave = False)

            for idx, (x, y) in loop_val:
                x = x.to(device = device)
                y = y.to(device = device)

                t_scores = model(x)

                t_loss = criterion(t_scores, y)
                t_losses.append(t_loss.item())

                _, t_predictions = t_scores.max(1)
                n_correct += (t_predictions==y).sum()
                n_samples += t_predictions.size(0)

                loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
                loop.set_postfix(loss = t_loss.item(), acc = float(n_correct)/float(n_samples)*100)
            # print(t_losses)
            test_loss = sum(t_losses)/len(t_losses)
            test_acc = float(n_correct)/float(n_samples) * 100

        # print(f"test loss at epoch {epoch} is {test_losss:.5f}, test accuracy is {test_acc}")

        print("Epoch: %d | Train Loss: %.4f | Train Accuracy: %.2f | Val Loss: %.4f | Val Accuracy: %.2f"  \
          %(epoch, sum(losses)/len(losses), float(num_correct)/float(num_samples)*100, test_loss, test_acc))

        # log the epoch train loss
        writer.add_scalar('training loss',
                    sum(losses)/len(losses),
                    epoch)
        # log the epoch train accuracy
        writer.add_scalar('training accuracy',
                        float(num_correct)/float(num_samples),
                        epoch)
        # log the epoch test loss
        writer.add_scalar('test loss',
                    test_loss,
                    epoch)
        # log the epoch test accuracy
        writer.add_scalar('test accuracy',
                        test_acc,
                        epoch)
        writer.close()

        if (epoch + 1) % 10 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, epoch, model_name = model_name)

# def dataset_prepare(data_dir, label_type = [0, 2], transform = None):

#     dataset = DEAP_Fnet_dataset(data_dir, label_type = label_type, transform = transform)
#     n_train = int(0.8*len(dataset))
#     n_test = len(dataset) - n_train
#     train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(29))
#     train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
#     test_loader = DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

#     return train_loader, test_loader

def dataset_prepare(data_dir, n_subjects = 26, channels = [2, 8, 9], label_type = [0, 2], input_length = 5, sampling_rate = 128, transform = None):
    train_data, train_labels, train_names, test_data, test_labels, test_names = load_DEAP(data_dir, n_subjects = n_subjects, label_type = label_type)
    n_repeat = 60//input_length

    train_data = train_data[:, channels, sampling_rate * 3:]
    train_data = train_data.reshape((-1, train_data.shape[1], input_length*sampling_rate))

    # print("test label: ", test_labels[:200])

    test_data = test_data[:, channels, sampling_rate * 3:]
    test_data = test_data.reshape((-1, test_data.shape[1], input_length*sampling_rate))

    train_labels = np.repeat(train_labels, n_repeat)
    test_labels = np.repeat(test_labels, n_repeat)
    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
    # print("test label: ", test_labels[:1000])
    train_set = DEAP_Fnet_dataset(train_data, train_labels, transform = transform)
    test_set = DEAP_Fnet_dataset(test_data, test_labels, transform = transform)

    # n_train = int(0.8*len(train_set))
    # n_test = len(train_set) - n_train
    # train_set, test_set = torch.utils.data.random_split(train_set, [n_train, n_test], generator=torch.Generator().manual_seed(29))
    train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    test_loader = DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

    return train_loader, test_loader, train_names, test_names

def get_random_dataset(train_size, test_size):
    train_data = torch.randn(train_size, 32, 640)
    y_train = torch.randint(0, 2, (train_size, ))
    test_data  = torch.randn(test_size, 32, 640)
    y_test = torch.randint(0, 2, (test_size, ))

    # train_data = torch.normal(0.453, 0,21, size = (train_size, 32, 640))
    # test_data = torch.normal(0.45, 0,23, size = (train_size, 32, 640))

    train_set = TensorDataset(train_data, y_train)
    test_set = TensorDataset(test_data, y_test)
    train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    test_loader = DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

    return train_loader, test_loader

def get_writer_path(write_root):
    if not os.path.exists(write_root):
        os.makedirs(write_root)
    existed_folds = os.listdir(write_root)
    write_name = os.path.join(write_root, f"v{len(existed_folds)+1}")

    return write_name

def save_info(write_root, test_names, train_names):

    data = np.array([train_names, test_names]).reshape(-1, 2)
    df = pd.DataFrame(data, columns = ['train', 'test'])
    df.to_csv(os.path.join(write_root, 'infor.csv'))

def get_model_name(attributes_list, model_name = ""):
    for attr in attributes_list:
        model_name = model_name + "-{}-{}".format(attr[0], attr[1])
    return model_name

if __name__=="__main__":
    for arg in vars(args):
        print (arg, getattr(args, arg))

    label_type = [0, 2]
    num_layers = 6
    d_model = 640
    expansion_factor = 2
    dropout = 0.2
    num_classes = label_type[1]
    print("num classes: ", num_classes)
    data_dir = r"D:\AIproject\emotion recognition\DEAP\data_preprocessed_python"
    channels = random.choices(np.arange(0, 32), k = 3)
    train_loader, test_loader, train_names, test_names = dataset_prepare(data_dir = data_dir, channels = channels, n_subjects = 26, label_type = label_type, transform = TRANSFORMS)
    print(channels)
    # i = 0
    # for d, t in train_loader:
    #     d = d.numpy()
    #     plt.figure(figsize= (10, 4))
    #     plt.plot(d[0, 1])
    #     plt.show()
    #     i+=1
    #     if i == 10:
    #         break
    # save_info(write_root, test_names, train_names)


    # model_name = f"FNet-d_model-{d_model}-ef-{expansion_factor}-nlayer-{num_layers}-dropout-{dropout}"
    # model = FNet(d_model=d_model, expansion_factor=expansion_factor, dropout=dropout, num_layers=num_layers, n_classes = num_classes).to(device)

    model_name = f"MLPmixer-d_model-{d_model}-ef-{expansion_factor}-nlayer-{num_layers}-dropout-{dropout}"
    model = MLPMixer(
        input_length = 5,
        channels = 3,
        patch_size = 1,
        sampling_rate = 128,
        dim = 512,
        depth = 10,
        num_classes = 2).to(device)

    # model.apply(init_weights)
    # model = CRNN(input_length = 5,
    #     channels = 3,
    #     patch_size = 1,
    #     sampling_rate = 128,
    #     dim = 512,
    #     num_classes = 2,
    #     num_layers = 2,
    #     hidden_size = 256).to(device)

    # attributes_list = model.get_attributes()
    # model_name = get_model_name(attributes_list, model_name = "RNN")
    # print(model_name)

    # optimizer = Adas(model.parameters(), lr = LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    write_root = get_writer_path(f"runs/{model_name}")
    writer = SummaryWriter(write_root)


    train_fn(model, criterion, optimizer, NUM_EPOCHS, writer, train_loader, test_loader, model_name)

    # Check accuracy on training & test to see how good our model

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)



    """
    model_name = "test_random"
    train_loader, test_loader = get_random_dataset(train_size = 10000, test_size = 1000)


    train_fn(model, criterion, optimizer, NUM_EPOCHS, writer, train_loader, test_loader, model_name)

    # Check accuracy on training & test to see how good our model

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
    """
