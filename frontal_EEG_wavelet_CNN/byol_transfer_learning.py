import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


from utils import dataset_prepare, dataset_prepare_v1, get_random_dataset
import numpy as np
from custom_transform import CustomTransform
import argparse
from byol_pytorch import BYOL

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--net', type=str, required = False, default = "effnet",
                       help='name of the base model')
parser.add_argument('--batch_size', type=int, required = False, default = 8,
                       help='batch size')
parser.add_argument('--n_epochs', type=int, required = False, default = 10,
                       help='number of epoch')

args = parser.parse_args()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 0) # valid convolution
        self.avgpool1 = nn.AvgPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 0) # valid convolution
        self.avgpool2 = nn.AvgPool2d((2, 2))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = (1, 1), padding = 0) # valid convolution
        self.avgpool3 = nn.AvgPool2d((2, 2))
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = 0) # valid convolution
        # self.avgpool4 = nn.AvgPool2d((2, 2))

        self.avgpool4 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, return_embedding = True):
        x = F.gelu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.gelu(self.conv2(x))
        x = self.avgpool2(x)
        # print(x.size())

        x = F.gelu(self.conv3(x))
        x = self.avgpool3(x)
        # print(x.size())

        x = F.gelu(self.conv4(x))
        x = self.avgpool4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)

        # print(x.size())
        return x

class BYOL_Transfer_Learning(pl.LightningModule):
    def __init__(self, net, n_classes, hidden_size, num_layers, dropout, layer = -2, epoch_to_unfree = -1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.epoch_to_unfree = epoch_to_unfree
        self.net = net
        # self.net = CNN()
        self.rnn = nn.LSTM(256, hidden_size, num_layers, batch_first=True) # 576

        # self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_classes),
        )

        self.accuracy = torchmetrics.Accuracy() # pl.metrics.Accuracy()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        if self.trainer.current_epoch < self.epoch_to_unfree:
            with torch.no_grad():
                c_out = self.net(c_in, return_embedding = True)
        else:
            c_out = self.net(c_in, return_embedding = True)
        # print(c_out)
        r_in = c_out.view(batch_size, timesteps, -1)
        # print(r_in.size())
        r_out, _ = self.rnn(r_in )
        # print(r_out.size())

        # # take all sequence output
        # out = r_out.reshape(r_out.shape[0], -1)

        # # take only last vector of sequence output
        out = r_out[:, -1, :].squeeze()

        out = self.mlp(out)
        # print(out)
        return out

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, min_lr=1e-7, eps=1e-08, verbose=True)
        return {
           'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'val_loss'
        }
        # return Adas(self.parameters(), lr = LR)

    def training_step(self, batch: dict, _batch_idx: int): # pylint: disable=arguments-differ
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(outputs.argmax(dim=1), labels))

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
    IMAGE_SIZE1 = 32
    IMAGE_SIZE2 = 128
    BATCH_SIZE = args.batch_size
    EPOCHS = args.n_epochs
    N_CLASSES = 2
    DATA_FILE_NAME = "EEG_frontal_segmented_6_2.npy"
    LABEL_FILE_NAME = "label_segmented_6_2.npy"
    ROOT_DIR = ""
    LABEL_TYPE = 0
    NUM_GPUS = 1
    scale = np.geomspace(0.8533, 9.6, num=IMAGE_SIZE1)
    wavelet_name = 'cgau1'
    TRANSFORMS = transforms.Compose([
        CustomTransform(
            scale = scale,
            wavelet_name= wavelet_name,
            frame_size = 1,
            overlap_size = 0.5,
            sampling_rate = 128,
            adding_noise = False,
        ),
    ])
    train_loader, val_loader = dataset_prepare(
            DATA_FILE_NAME,
            LABEL_FILE_NAME,
            ROOT_DIR,
            BATCH_SIZE,
            LABEL_TYPE,
            TRANSFORMS,
        )

    # train_loader, val_loader = get_random_dataset(
    #     batch_size = BATCH_SIZE,
    #     train_size = 1000,
    #     test_size = 200,
    #     shape = [2, 1, IMAGE_SIZE1, IMAGE_SIZE2]
    # )
    # # create model and trainer
    effnet = EfficientNet.from_name('efficientnet-b0')
    effnet._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size = (IMAGE_SIZE1, IMAGE_SIZE2))
    byol = BYOL(
        effnet,
        image_size1 = IMAGE_SIZE1,
        image_size2 = IMAGE_SIZE2,
        channels = 1,
        hidden_layer = '_avg_pooling', # use "avgpool" to get representation of torch models resnet50
        use_momentum = False,
        projection_size = 256,
        projection_hidden_size = 512,
        # moving_average_decay = 0.99
    )

    try:
        print("loading model")
        byol.load_state_dict(torch.load(f'./weights/byol-effnet-b0-improved-net.pt'))
        print("load model done!")
        # for param in byol.parameters():
        #     print(param)
    except:
        print("model load failed!")

    model = BYOL_Transfer_Learning(
        byol,
        n_classes = N_CLASSES,
        hidden_size = 256,
        num_layers = 2,
        dropout = 0.2,
        layer = "_avg_pooling",
        epoch_to_unfree = 10,
    )

    # x = torch.randn(1, 5, 1, 128, 128)
    # print(model(x))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_weights_only=True,
        verbose=True,
        dirpath=f'weights/{args.net}-{IMAGE_SIZE1}x{IMAGE_SIZE2}',
        filename= f"{args.net}"+"-{epoch:02d}-{val_loss:.2f}.ckpt"
    )
    tb_logger = pl_loggers.TensorBoardLogger(name = f"{args.net}-{IMAGE_SIZE1}x{IMAGE_SIZE2}-{N_CLASSES}", save_dir = 'lightning_logs')

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 4,
        # auto_lr_find=True,
        callbacks = [checkpoint_callback],
        logger=tb_logger,
        # val_check_interval=0.25,
        check_val_every_n_epoch = 1,
        precision=16,
        # resume_from_checkpoint="",
        # default_save_path = './weights'
    )

    trainer.fit(model, train_loader, val_loader)


# test dataloader

    # import matplotlib.pyplot as plt

    # for batch in train_loader:
    #     x, y = batch
    #     print(x)
    #     print(y)
    #     for s in x:
    #         fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    #         fig.subplots_adjust(hspace = .5, wspace=.001)

    #         axs = axs.ravel()

    #         for i in range(10):

    #             axs[i].imshow(s[i].squeeze(), cmap = 'jet')
    #             axs[i].set_title(f"step {i}")

    #         plt.show()


    #     break

