from scipy.sparse.construct import random
import torch 
import torch.nn as nn
import torch.functional as F
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from einops import rearrange, repeat
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import dataset_prepare, dataset_prepare_for_KF
from sklearn.model_selection import KFold
from transformer import Transformer

from custom_dataset import DEAP_dataset
from custom_transforms import CustomTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
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

class ChannelWiseAttention(nn.Module):
    def __init__(self, n_channels, hidden_size):
        super().__init__()
        # first projection layer, bias is True by default 
        self.linear_in = nn.Linear(n_channels, hidden_size)
        # output projection layer, bias is True by default
        self.linear_out = nn.Linear(hidden_size, n_channels)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, x):
        # input size: (batch_size, n_steps, n_channel, n_samples)
        
        # calculate attention map
        # s_shape = (batch_size, n_steps, n_channel)
        s = torch.mean(x, dim = -1)
        # s_shape = (batch_size, n_steps, hidden_size)
        s = self.linear_in(s)
        # s_shape = (batch_size, n_steps, n_channel)
        s = self.tanh(self.linear_out(s))
        attention_map = self.softmax(s)
        # attention_map shape = (batch_size, n_steps, n_channel, 1)
        attention_map = attention_map.unsqueeze(-1)
        # calculate output: cj = aj * xj
        out = x * attention_map
        
        return out

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels = 40, 
            kernel_size=(N_SCALE, 40), 
        )
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 75), stride=10)

    def forward(self, x):
        # input shape: (batch_size, n_steps, n_channels, n_samples)
        # reshape to do parallel convolution on every step
        # new_shape = (batch * n_step, 1, c, n)
        batch, step, c, h, w = x.shape
        x = x.view(batch * step, c, h, w) 
        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        out = x.view(batch, step, -1) 

        return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

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
    def __init__(self, feature_dim, num_classes, dim, depth, heads, mlp_dim,  n_steps, eeg_channels, cwa_hidden, cnn_in_channels, dim_head = 64, pool = "cls", dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.rand(1, n_steps + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # model architecture
        self.cwa = ChannelWiseAttention(n_channels=eeg_channels, hidden_size=cwa_hidden)
        self.cnn = CNN(cnn_in_channels)
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
        x = x.view(b, n, c, h*w)
        x = self.cwa(x)
        x = x.view(b, n, c, h, w)
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
        # print(labels.shape, out/puts.shape)
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
    feature_dim = 80 # this is # output features from CNN 
    num_classes = 2
    dim = 128
    depth = 2
    heads = 4
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
        n_steps, eeg_channels, 
        cwa_hidden,
        cnn_in_channels, 
        dim_head = 64, 
        pool = "cls", 
        dropout = dropout, 
        emb_dropout = emb_dropout,
    ).to(DEVICE)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_weights_only=True,
        verbose=True,
        dirpath=f'weights/ACTransformer_sub_all_subjects',
        filename= f"ACTransformer"+"-{epoch:02d}-{val_loss:.2f}.ckpt"
    )
    tb_logger = pl_loggers.TensorBoardLogger(name = f"ACTransformer_sub_all_subjects", save_dir = 'lightning_logs')

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
        # resume_from_checkpoint="",
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
