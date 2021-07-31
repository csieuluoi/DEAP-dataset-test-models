from scipy.sparse.construct import random
import torch 
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.container import ModuleList
import torchmetrics

from efficient_channel_attention import eca_layer, ChannelWiseAttention

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from einops import rearrange, repeat
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import dataset_prepare, dataset_prepare_for_KF
from sklearn.model_selection import KFold
from transformer import Transformer, Attention

from custom_dataset import DEAP_dataset
from custom_transforms import CustomTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 40
NUM_GPUS = 1
N_SUBJECT = 1
LR = 1e-3
N_SCALE = 8
TRANSFORM = CustomTransform(
    scale = None, 
    n_scale = N_SCALE, 
    wavelet = "morl", 
    sampling_rate = 128, 
    normalized=True
)


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size=(1, 75), pooling_stride = 10):
        super().__init__()

        self.attn = Attention(dim = 128, heads = 1, dim_head = 64, dropout = 0.)
        self.conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            )
        self.pool = nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride)
        
        self.attn1 = Attention(dim = 64, heads = 1, dim_head = 64, dropout = 0.)
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = out_channels,
            kernel_size=(3, 3),
            stride = 2,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels*2,
            kernel_size=(3, 3)
        )
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gelu = nn.GELU()



    def forward(self, x):
        # input shape: (batch_size, n_steps, n_channels, n_samples)
        # reshape to do parallel convolution on every step
        # new_shape = (batch * n_step, 1, c, n)
        batch, step, c, h, w = x.shape

        x = x.view(batch * step* c, h, w) 
        x = self.attn(x)

        x = x.view(batch * step * c, 1, h, w) 
        x = self.conv(x)
        x = self.gelu(x)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(batch * step, c, -1) 
        # print(x.shape)

        x = self.attn1(x)
        # print(x.shape)
 
        x = x.view(batch * step, 1, c, -1)
        # print(x.shape)

        x = self.conv1(x)
        x = self.gelu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.gelu(x)
        # print(x.shape)
        x = self.pool2(x)
        
        out = x.view(batch, step, -1)

        return out



# class ACNN_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss = nn.CrossEntropyLoss()

    
#     def get_labels(self, preds, target):
#         # print(target.shape)
#         labels = target.unsqueeze(-1)
#         labels = labels.expand(preds.size()[0], preds.size()[1])
#         # print(labels.shape)
#         return labels
    
#     def __call__(self, preds, target_is_real):
#         losses = 0
#         labels = self.get_labels(preds, target_is_real)
    
#         for i in range(preds.shape[1]):
#             # print(preds[:, i, :].squeeze().shape, labels[:, i].shape)
#             losses += self.loss(preds[:, i, :].squeeze(), labels[:, i])
#         # losses = losses/preds.shape[1]
#         return losses

# class ARCNN_accuracy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.accuracy = pl.metrics.Accuracy()

#     def get_labels(self, preds, target):
#         # print(target.shape)
#         labels = target.unsqueeze(-1)
#         labels = labels.expand(preds.size()[0], preds.size()[1])
#         # print(labels.shape)
#         return labels
    
#     def __call__(self, preds, target_is_real):
#         accuracies = 0
#         labels = self.get_labels(preds, target_is_real)
    
#         for i in range(preds.shape[1]):
#             # print(preds[:, i, :].squeeze().shape, labels[:, i].shape)
#             accuracies += self.accuracy(preds[:, i, :].squeeze(), labels[:, i])
#         accuracies = accuracies/preds.shape[1]
#         return accuracies


class ACTransformer(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, dim, depth, heads, mlp_dim, n_steps, dim_head = 64, pool = "cls", dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.save_hyperparameters()

        self.pos_embedding = nn.Parameter(torch.rand(1, n_steps + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # model architecture
        self.cnn =  CNN(1, 32, (8, 40)) # this output ( _, steps, 64)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_embedding = nn.Linear(feature_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
		# self.to_cls_token = nn.Identity()
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim, mlp_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(mlp_dim, num_classes)
            nn.Linear(dim, num_classes)

        )


        # self.train_set, self.val_set = dataset_prepare(n_subjects=N_SUBJECT)
        self.accuracy = pl.metrics.Accuracy()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, x, mask = None):
        b, n, c, h, w = x.shape
        x = self.cnn(x)

        x = self.to_embedding(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # print(x.shape, cls_tokens.shape)

        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n+1)]

        x = self.dropout(x)

        x = self.transformer(x, mask)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        output = self.mlp_head(x)

        return output

    
    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(dataset = self.train_set, batch_size = BATCH_SIZE, shuffle = True)

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(dataset = self.val_set, batch_size = BATCH_SIZE)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        return torch.optim.Adam(self.parameters(), lr = LR)

    def training_step(self, batch, _batch_idx): # pylint: disable=arguments-differ
        inputs, labels = batch
        outputs = self(inputs)
        # print(labels.shape, outputs.shape)
        loss = self.crossEntropyLoss(outputs, labels)
        
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(outputs, labels))

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.crossEntropyLoss(logits, y)
        acc = self.accuracy(logits, y)

        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

if __name__ == "__main__":
    feature_dim = 64 # this is # output features from CNN 
    num_classes = 2
    dim = 64
    depth = 4
    heads = 6
    mlp_dim = 128
    n_steps = 3
    eeg_channels = 32
    cwa_hidden = 128
    cnn_in_channels = 32
    dropout = 0.0
    emb_dropout = 0.0

    model = ACTransformer(
        feature_dim,
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim,  
        n_steps,
        dim_head = 64, 
        pool = "cls", 
        dropout = dropout, 
        emb_dropout = emb_dropout,
    ).load_from_checkpoint("weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=35-val_loss=0.58.ckpt.ckpt")

    # model.load_from_checkpoint("weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_weights_only=True,
        verbose=True,
        dirpath=f'weights/Bi-ATransformer_sub_all_subjects',
        filename= f"Bi-ATransformer"+"-{epoch:02d}-{val_loss:.2f}.ckpt"
    )
    tb_logger = pl_loggers.TensorBoardLogger(name = f"Bi-ATransformer_sub_all_subjects", save_dir = 'lightning_logs')

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        # accumulate_grad_batches = 1,
        # auto_lr_find=True,
        callbacks = [checkpoint_callback],
        logger=tb_logger,
        # val_check_interval=0.25,
        check_val_every_n_epoch = 1,
        precision=16,
        # resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
        # default_save_path = './weights'
    )
   

    """Training code"""
    # # # all subjects leave some subjects out
    X_train, y_train, X_test, y_test = dataset_prepare(
        segment_duration = 3, 
        load_all = True, 
        return_dataset = False, 
        sampling_rate = 128
    )
    print("train x shape:", X_train.shape)
    print("train y shape:", y_train.shape)

    train_set = DEAP_dataset(data = X_train, labels = y_train, transform = TRANSFORM)
    val_set = DEAP_dataset(data = X_test, labels = y_test, transform = TRANSFORM)

    train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True) 
    val_loader = DataLoader(dataset = val_set, batch_size = BATCH_SIZE, shuffle = False)

    trainer.fit(model, train_loader, val_loader)

    
    # # perform k-fold:
    # for nth_subject in range(32):
    #     X, y = dataset_prepare_for_KF(n_subjects=nth_subject)
    #     kf = KFold(10, shuffle = True, random_state = 29)

    #     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    #         print("training fold: ", fold+1)
    #         X_train, X_val = X[train_idx], X[val_idx]
    #         y_train, y_val = y[train_idx], y[val_idx]

    #         train_x = torch.Tensor(X_train) # transform to torch tensor
    #         train_y = torch.Tensor(y_train)
    #         val_x = torch.Tensor(X_val) # transform to torch tensor
    #         val_y = torch.Tensor(y_val)

    #         train_set = TensorDataset(train_x, train_y.long()) # create your datset
    #         val_set = TensorDataset(val_x, val_y.long())
    #         train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
    #         val_loader = DataLoader(dataset = val_set, batch_size = BATCH_SIZE, shuffle = False)

    #         model = ACRNN(
    #         n_classes = 2, 
    #         eeg_channels = 32,
    #         cwa_hidden = 64, 
    #         cnn_in_channels = 1, 
    #         rnn_hidden = 64, 
    #         rnn_embed_dim = 128,
    #         ).to(DEVICE)

    #         checkpoint_callback = ModelCheckpoint(
    #             monitor='val_loss',
    #             save_weights_only=True,
    #             verbose=True,
    #             dirpath=f'weights/subject_{nth_subject}/ACRNN_fold_{fold+1}',
    #             filename= f"ACRNN"+"-{epoch:02d}-{val_loss:.2f}.ckpt"
    #         )
    #         tb_logger = pl_loggers.TensorBoardLogger(name = f"ACRNN_fold_{fold+1}", save_dir = f'lightning_logs/subject_{nth_subject}')

    #         trainer = pl.Trainer(
    #             gpus = NUM_GPUS,
    #             max_epochs = EPOCHS,
    #             # accumulate_grad_batches = 1,
    #             # auto_lr_find=True,
    #             callbacks = [checkpoint_callback],
    #             logger=tb_logger,
    #             # val_check_interval=0.25,
    #             check_val_every_n_epoch = 1,
    #             precision=16,
    #             # resume_from_checkpoint="",
    #             # default_save_path = './weights'
    #         )

    #         trainer.fit(model, train_loader, val_loader)
