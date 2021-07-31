import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


from pathlib import Path
from typing import Dict, List, Union
from einops.layers.torch import Rearrange, Reduce

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), stride = (2, 2), padding = 0) # valid convolution
        self.avgpool1 = nn.AvgPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (2, 2), padding = 0) # valid convolution
        self.avgpool2 = nn.AvgPool2d((2, 2))
        # self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = (1, 1), padding = 0) # valid convolution
        # self.avgpool3 = nn.AvgPool2d((2, 2))
        # self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = 0) # valid convolution
        # self.avgpool4 = nn.AvgPool2d((2, 2))

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.gelu(self.conv2(x))
        x = self.avgpool2(x)
        # print(x.size())

        # x = F.gelu(self.conv3(x))
        # x = self.avgpool3(x)
        # print(x.size())

        # x = F.gelu(self.conv4(x))
        # x = self.avgpool4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)

        # print(x.size())
        return x



class CRNN(pl.LightningModule):
    instances = []

    def __init__(self, input_length, channels, patch_size, sampling_rate, dim, num_classes, num_layers, hidden_size, learning_rate=1e-3):
        super(CRNN, self).__init__()
        self.__class__.instances.append(self)

        n_patches = (input_length) // (patch_size)
        self.patch_size = patch_size
        self.sampling_rate = sampling_rate
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learningRate = learning_rate
        self.accuracy = pl.metrics.Accuracy()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        # self.cnn = CNN()

        self.rnn = nn.LSTM(patch_size * sampling_rate * channels, hidden_size, num_layers, batch_first=True) # 576
        # self.rnn2 = nn.LSTM(256, hidden_size, 1, batch_first=True) # 576

        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.dropout1 = nn.Dropout(0.1)

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():

            print(attribute, '=', value)

    def get_attributes(self):
        return list(self.__dict__.items())[-6:]

    def forward(self, x):
        x = Rearrange('b c (n p s) -> b n (p s c)', p = self.patch_size, s = self.sampling_rate)(x)
        # print(x.size())
        # initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # (1, batch_size, 128)
        # initialize cell state for LSTM
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # (1, batch_size, 128)

        r_in = x
        # print(r_in.size())
        r_out, _ = self.rnn(r_in , (h0, c0))
        # print(r_out.size())

        # take all sequence output
        # out = r_out.reshape(r_out.shape[0], -1)

        # take only last vector of sequence output
        out = r_out[:, -1, :].squeeze()
        # print(out.size())
        out = self.dropout1(out)
        out = self.fc1(out)

        return out

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)

    def training_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]: # pylint: disable=arguments-differ
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(outputs, labels))

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.crossEntropyLoss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_acc', acc)
        return loss
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')





if __name__ == "__main__":
    model = CRNN(input_length = 5,
        channels = 32,
        patch_size = 1,
        sampling_rate = 128,
        dim = 512,
        num_classes = 2,
        num_layers = 2,
        hidden_size = 256).to(device)
    # summary(model, (60, 1, 32, 32))
    model.print_instance_attributes()
    x = torch.randn(16, 32, 640).to(device)
    print(model(x).shape)
