import torch
import torch.nn as nn
import torch_geometric
from einops import rearrange

import pytorch_lightning as pl
import torchmetrics
from einops.layers.torch import Rearrange

EEG_channels_name = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "Fp2",
    "AF4",
    "Fz",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "Cz",
    "C4",
    "T8",
    "P6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]
g1 = ["Fp1", "AF7", "AF3"]
g2 = ["F7", "F5", "F3", "F1"]
g3 = ["FC5", "FC3", "FC1"]
g4 = ["Fp2", "AF4", "AF8"]
g5 = ["F2", "F4", "F6", "F8"]
g6 = ["FC2", "FC4", "FC6"]
g7 = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"]
g8 = ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
g9 = ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]
g10 = ["PO7", "PO3", "POz", "PO4", "PO8"]
g11 = ["O1", "Oz", "O2"]

DEVICE = torch.device("cuda:0")


def get_channel_index(channel_name_list, all_channels=EEG_channels_name):
    idxes = []
    for name in channel_name_list:
        if name in all_channels:
            idxes.append(all_channels.index(name))

    return idxes


G_IDX = [
    get_channel_index(g1),
    get_channel_index(g2),
    get_channel_index(g3),
    get_channel_index(g4),
    get_channel_index(g5),
    get_channel_index(g6),
    get_channel_index(g7),
    get_channel_index(g8),
    get_channel_index(g9),
    get_channel_index(g10),
    get_channel_index(g11),
]


def get_kernel_size(k):
    return int((0.5**k) * 128)


"""TSception is the CNN1D in LGG net"""


class TCNN1D(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel,
                stride=step,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(self, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: EEG channel x datapoint
        super(TCNN1D, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(
            1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool
        )
        self.Tception2 = self.conv_block(
            1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool
        )
        self.Tception3 = self.conv_block(
            1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool
        )

        self.Sception1 = self.conv_block(num_T, num_S, (1, 1), 1, int(self.pool * 0.25))

        self.BN_t = nn.BatchNorm2d(num_T)

        self.size = self.get_size(input_size)

    def forward(self, x):
        ## original code: x : (batch, channel, n)
        ## code from tsception repo: x : (batch, 1, channel, n)
        # b, c, n = x.shape
        # x = rearrange(x, "b c n -> b 1 c n")
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)

        out = self.Sception1(out)
        out = out.squeeze(1)
        return out

    def get_size(self, input_size):
        # here we use an array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        x = torch.ones((1, 1, input_size[-2], int(input_size[-1])))
        # b, c, n = x.shape
        # x = rearrange(x, "b c n -> b 1 c n")
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)

        out = self.Sception1(out)
        out = out.squeeze(1)

        return out.size()


class LocalGNN(nn.Module):
    def __init__(self, input_size, g_idx=G_IDX):
        super().__init__()

        self.local_W = nn.Parameter(torch.FloatTensor(input_size[-2], input_size[-1]))
        self.local_b = nn.Parameter(torch.FloatTensor(input_size[-2], 1))

        torch.nn.init.xavier_normal_(self.local_W)
        torch.nn.init.xavier_normal_(self.local_b)

        pool = 8
        self.relu = nn.ReLU()
        # self.avgpool = nn.AvgPool1d(kernel_size = pool)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))
        self.g_idx = g_idx
        self.size = self.get_size(input_size)

    def forward(self, x):
        Z_filtered = self.avgpool(self.relu(x * self.local_W - self.local_b))
        Z_local = torch.empty((Z_filtered.shape[0], 11, Z_filtered.shape[-1])).to(
            Z_filtered.get_device()
        )

        for i in range(11):
            z_m = torch.mean(Z_filtered[:, self.g_idx[i], :], dim=1)
            Z_local[:, i, :] = z_m.squeeze()
        return Z_local

    def get_size(self, input_size):
        x = torch.ones((1, input_size[-2], int(input_size[-1])))

        Z_filtered = self.avgpool(self.relu(x * self.local_W - self.local_b))
        Z_local = torch.empty((Z_filtered.shape[0], 11, Z_filtered.shape[-1]))
        for i in range(11):
            z_m = torch.mean(Z_filtered[:, self.g_idx[i], :], dim=1)
            Z_local[:, i, :] = z_m.squeeze()

        return Z_local.size()


class GlobalGNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_size[1])

        self.global_A = nn.Parameter(
            torch.FloatTensor(input_size[-2], input_size[-2])
        )  # , requires_grad = True,  device=DEVICE)
        ## initialize weight of adjacency matrix A according to xavier initialization
        torch.nn.init.xavier_normal_(self.global_A)
        self.global_W = nn.Linear(input_size[-1], hidden_size)

        self.relu = nn.ReLU()
        self.flatten = Rearrange("b c s -> b (c s)")

        self.size = self.get_size(input_size)

    def forward(self, x):

        x = self.batchnorm(x)
        x = torch.matmul(
            (self.relu(self.global_A + self.global_A.transpose(0, 1)) * 1 / 2), x
        )

        x = self.relu(self.global_W(x))
        x = self.flatten(x)

        return x

    def get_size(self, input_size):
        # print(input_size)
        x = torch.ones((1, input_size[-2], int(input_size[-1])))
        # print(x.shape)
        # print(self.global_A.shape)

        x = self.batchnorm(x)
        x = torch.matmul(self.global_A, x)

        x = self.relu(self.global_W(x))
        x = self.flatten(x)

        return x.size()


class LGGNet(pl.LightningModule):
    def __init__(
        self,
        input_size=(1, 32, 7680),
        num_classes=2,
        T_kernels=5,
        hidden=32,
        dropout_rate=0.3,
    ):
        super().__init__()
        # self.device = device
        # self.TConv = TCNN1D((32,7680),128,T_kernels,1,128,0.2)
        self.TConv = TCNN1D(input_size, 128, T_kernels, 1, 128, 0.2)
        self.lgnn = LocalGNN(input_size=self.TConv.size)
        self.ggnn = GlobalGNN(input_size=self.lgnn.size)

        self.fc = nn.Sequential(
            nn.Linear(self.ggnn.size[-1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        )

        self.crossEntropyLoss = nn.CrossEntropyLoss()

        acc = torchmetrics.Accuracy()
        # use .clone so that each metric can maintain its own state
        self.train_acc = acc.clone()
        # assign all metrics as attributes of module so they are detected as children
        self.valid_acc = acc.clone()

    def forward(self, x):
        x = self.TConv(x)
        x = self.lgnn(x)
        x = self.ggnn(x)
        x = self.fc(x)

        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)

        loss = self.crossEntropyLoss(outputs, y)
        # print(outputs, "/n", y)
        self.log("train_loss", loss)

        return {"loss": loss, "preds": outputs, "targets": y}

    def training_step_end(self, outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.train_acc(outs["preds"], outs["targets"].int())
        self.log("train/acc_step", self.train_acc)

    def training_epoch_end(self, outs):
        # additional log mean accuracy at the end of the epoch
        self.log("train/acc_epoch", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # print(outputs.shape, y.shape)
        loss = self.crossEntropyLoss(outputs, y)

        self.log("val_loss", loss)

        return {"preds": outputs, "targets": y}

    def validation_step_end(self, outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.valid_acc(outs["preds"], outs["targets"].int())
        self.log("val/acc_step", self.valid_acc)

    def validation_epoch_end(self, outs):
        # additional log mean accuracy at the end of the epoch
        self.log("val/acc_epoch", self.valid_acc.compute())


if __name__ == "__main__":
    # model = TCNN1D((32,7680),128,5,1,128,0.2)

    model = LGGNet(num_classes=2).to("cuda")
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    x = torch.randn(2, 1, 32, 7680).to("cuda")
    out = model(x)
    print(out)

    targets = torch.LongTensor([0, 1]).to("cuda")
    loss_fn = nn.CrossEntropyLoss()

    loss = loss_fn(out, targets)
    print(loss)
    ### test small modules
    # tcnn1d = TCNN1D((1, 32, 512),128,5,1,128,0.2).to("cuda")
    # x = torch.randn(2, 1, 32, 512).to("cuda")
    # out = tcnn1d(x)
    # # print(out.shape)

    # lgnn = LocalGNN(input_size = tcnn1d.size).to("cuda")
    # out = lgnn(out)
    # out = out.to("cuda")
    # ggnn = GlobalGNN(input_size = lgnn.size).to("cuda")
    # out = ggnn(out)
    # print(out.shape)
