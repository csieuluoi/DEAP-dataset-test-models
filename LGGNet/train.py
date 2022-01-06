import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from model import LGNNet

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from imblearn.over_sampling import RandomOverSampler
from utils import get_one_subject, get_dataloader

from einops import rearrange
import torch
import torch.nn as nn

DEVICE = "cuda"
NUM_GPUS = 1
SUBJECT_NUMBER = 1
EPOCHS = 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


if __name__ == "__main__":
	BATCH_SIZE = 8
	## get data of one subject
	X, y, _ = get_one_subject(SUBJECT_NUMBER)

	## implement leave one trial out
	loo = LeaveOneOut()
	loo.get_n_splits(X)

	for train_index, val_index in loo.split(X):
		print("TRAIN:", train_index, "VAL:", val_index)
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]

		## duplicate minority label
		X_train = rearrange(X_train, "n c s -> n (c s)")
		oversample = RandomOverSampler(sampling_strategy='minority')
		X_train, y_train = oversample.fit_resample(X_train, y_train)
		print(y_train.shape)
		print(X_train.shape)
		X_train = rearrange(X_train, "n (c s) -> n c s", c = 32)

		## implement k-fold
		kf = KFold(n_splits=5)
		kf.get_n_splits(X_train)
		for fold, (train_fold_index, test_index) in enumerate(kf.split(X_train)):
			X_train_fold, X_test = X_train[train_fold_index], X_train[test_index]
			y_train_fold, y_test = y_train[train_fold_index], y_train[test_index]

			train_loader = get_dataloader(X_train_fold, y_train_fold, BATCH_SIZE, shuffle = True)
			test_loader = get_dataloader(X_test, y_test, BATCH_SIZE, shuffle = False)

			##  initialize the model and trainer
			model = LGNNet()
			model.apply(init_weights)
			model.to(DEVICE)

			checkpoint_callback = ModelCheckpoint(
				monitor="val_loss",
				save_weights_only=True,
				verbose=True,
				dirpath=f"weights/subject_{SUBJECT_NUMBER:02d}",
				filename=f"Fold={fold+1}" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
			)
			# tb_logger = pl_loggers.TensorBoardLogger(
			#     name=f"{model_name}_sub_all_subjects", save_dir="lightning_logs"
			# )

			wandb_logger = pl_loggers.WandbLogger(project = "LGGNET")

			trainer = pl.Trainer(
				gpus=NUM_GPUS,
				max_epochs=EPOCHS,
				# accumulate_grad_batches = 1,
				# auto_lr_find=True,
				callbacks=[checkpoint_callback],
				logger=wandb_logger,
				# val_check_interval=0.25,
				check_val_every_n_epoch=1,
				# precision=32,
				# resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
				# default_save_path = './weights'
			)

			trainer.fit(model, train_loader, test_loader)

			break
		break
