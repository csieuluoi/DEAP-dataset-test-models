import torch
import torch_geometric
from einops import rearrange

import pytorch_lightning as pl

EEG_channels_name = ["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1","Oz","Pz","Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","P6","CP2","P4","P8","PO4","O2"]
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

def get_channel_index(channel_name_list, all_channels = EEG_channels_name):
    idxes = []
    for name in channel_name_list:
        if name in all_channels:
            idxes.append(all_channels.index(name))

    return idxes

def get_kernel_size(k):
    return int((0.5**k) * 128)

class CNN1D(torch.nn.Module):
    def __init__(self, T=5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, T, get_kernel_size(1))
        self.conv2 = torch.nn.Conv1d(1, T, get_kernel_size(2))
        self.conv3 = torch.nn.Conv1d(1, T, get_kernel_size(3))

        self.relu = torch.nn.LeakyReLU(0.01)
        self.avgpool1 = torch.nn.AvgPool1d(kernel_size = get_kernel_size(1))
        self.avgpool2 = torch.nn.AvgPool1d(kernel_size = get_kernel_size(2))
        self.avgpool3 = torch.nn.AvgPool1d(kernel_size = get_kernel_size(3))

        self.batchnorm = torch.nn.BatchNorm1d(T)
        self.conv11 = torch.nn.Conv1d(T, 1, 1)

    def forward(self, x):
        b, c, n = x.shape

        x1 = rearrange(x, "b c n -> (b c) 1 n")
        x1 = self.avgpool1(self.relu(self.conv1(x1)))

        x2 = rearrange(x, "b c n -> (b c) 1 n")
        x2 = self.avgpool2(self.relu(self.conv1(x2)))

        x3 = rearrange(x, "b c n -> (b c) 1 n")
        x3 = self.avgpool3(self.relu(self.conv1(x3)))

        x = torch.concat((x1, x2, x3), dim = -1)
        x = self.batchnorm(x)

        x = self.conv11(x)
        x = rearrange(x, "(b c) c1 n -> b c c1 n", c = c)
        x = x.squeeze()

        return x



class LocalGNN(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.local_W = torch.nn.Parameter(torch.randn(input_shape[1], input_shape[2]))
        self.local_b = torch.nn.Parameter(torch.randn(input_shape[1], 1))

        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool1d(kernel_size = 3)
        self.g_idx = [
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

    def forward(self, x, device):
        Z_filtered = self.avgpool(self.relu(x*self.local_W - self.local_b))
        Z_local = torch.empty((Z_filtered.shape[0], 11, Z_filtered.shape[-1]), device = device)

        for i in range(11):
            z_m = torch.mean(Z_filtered[:, self.g_idx[i], :], dim = 1)
            Z_local[:, i, :] = z_m.squeeze()

        return Z_local

class GlobalGNN(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm1d(input_shape[1])

        self.global_A = torch.nn.Parameter(torch.randn(input_shape[1], input_shape[1]))
        self.global_W = torch.nn.Parameter(torch.randn(input_shape[2], 64))
        self.global_b = torch.nn.Parameter(torch.randn(input_shape[1], 1))

        self.relu = torch.nn.ReLU()
        self.linear_out = torch.nn.Linear(704, 2)
        self.dropout = torch.nn.Dropout(p=0.3)


    def forward(self, x, device):
        x = self.batchnorm(x)
        x = torch.matmul(self.global_A, x)

        x = torch.matmul(x, self.global_W)
        x = self.relu(x - self.global_b)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_out(self.dropout(x))

        return self.sigmoid(x)


class LGNNet(pl.LightningModule):
    def __init__(self, T_kernels = 5):
        super().__init__()

        self.TConv = CNN1D(T = T_kernels)
        self.lgnn = LocalGNN(input_shape = (1, 32, 833))
        self.ggnn = GlobalGNN(input_shape = (1, 11, 277))

        self.bceloss = nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x, device):
        x = self.TConv(x)
        x = self.lgnn(x, device)
        x = self.ggnn(x, device)

        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2, momentum = 0.9)

    def training_step(self, batch):
        x, y = batch
        outputs = self(x)

        loss = self.bceloss(outputs, labels)

        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(outputs, labels))

        return loss

    def validation_step(self, batch):
        x, y = batch
        outputs = self(x)

        loss = self.bceloss(outputs, labels)

        self.log("validation_loss", loss)
        self.log("validation_acc", self.accuracy(outputs, labels))

        return loss


if __name__=="__main__":

    """====================================================================="""
    """========================== test CNN1D ==============================="""
    # model = CNN1D(in_channels = 32)
    # x = torch.randn(8, 32, 7680)
    # out = model(x)
    # print(out.shape)

    """====================================================================="""
    """========================== test LGNN ================================"""
    # x = torch.randn(8, 32, 833)

    # model = LocalGNN(input_shape = x.shape)
    # out = model(x, "cuda")
    # print(model.local_W.shape)
    # print(out.shape)
    # print(out)

    # print(len(g1) + len(g2) + len(g3) + len(g4) + len(g5) + len(g6) + len(g7) + len(g8) + len(g9)+len(g10)+len(g11))

    # print([EEG_channels_name.index(c) for c in g1])
    # idxes = get_channel_index(g9, EEG_channels_name)

    # print(idxes)
    # print([EEG_channels_name[idx] for idx in idxes])
    # print(g9)

    # x = torch.randn(8, 11, 277)
    # model = GlobalGNN(x.shape)
    # device = "cuda"
    # out = model(x, device)

    # print(out.shape)
    x = torch.randn(8, 32, 7680).to("cuda")
    net = LGNNet().to("cuda")

    out = net(x, "cuda")
