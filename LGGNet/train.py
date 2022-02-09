import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# from model import LGNNet
from model_new_code import LGNNet, TSception

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from imblearn.over_sampling import RandomOverSampler
from utils import get_one_subject, get_dataloader

from einops import rearrange
import torch
import torch.nn as nn
import os
import numpy as np

DEVICE = torch.device("cuda:0")
NUM_GPUS = 1
EPOCHS = 25

def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0.01)

	if type(m) == nn.Conv2d:
		torch.nn.init.xavier_normal_(m.weight)


def get_best_weight(weight_dir = ""):
	file_names = os.listdir(weight_dir)
	losses = []
	for name in file_names:
		loss = float(name.split("=")[3][:4])
		losses.append(loss)

	return file_names[np.argmin(losses)], int(file_names[np.argmin(losses)].split("=")[2][:2])

if __name__ == "__main__":
	BATCH_SIZE = 8

	# get data of one subject
	for SUBJECT_NUMBER in range(2, 33):
		X, y, _ = get_one_subject(SUBJECT_NUMBER)

		## implement leave one trial out
		loo = LeaveOneOut()
		loo.get_n_splits(X)
		predictions = []
		targets = []
		for train_index, val_index in loo.split(X):
			print("TRAIN:", train_index, "VAL:", val_index)
			X_train, X_val = X[train_index], X[val_index]
			y_train, y_val = y[train_index], y[val_index]

			## duplicate minority label
			X_train = rearrange(X_train, "n c s -> n (c s)")
			oversample = RandomOverSampler(sampling_strategy='minority')
			X_train, y_train = oversample.fit_resample(X_train, y_train)
			print(y_train.shape)
			X_train = rearrange(X_train, "n (c s) -> n c s", c = 32)
			print(X_train.shape)

			## implement k-fold
			weight_dir = f"weights/LGGNet_subject_{SUBJECT_NUMBER:02d}/LOTO_index_{val_index[0]}"
			kf = KFold(n_splits=5)
			kf.get_n_splits(X_train)
			for fold, (train_fold_index, test_index) in enumerate(kf.split(X_train)):
				X_train_fold, X_test = X_train[train_fold_index], X_train[test_index]
				y_train_fold, y_test = y_train[train_fold_index], y_train[test_index]

				train_loader = get_dataloader(X_train_fold, y_train_fold, BATCH_SIZE, shuffle = True)
				test_loader = get_dataloader(X_test, y_test, BATCH_SIZE, shuffle = False)

				##  initialize the model and trainer
				model = LGNNet().to(DEVICE)
				# model.apply(init_weights)
				# model.to(DEVICE)

				# print(model.lgnn.local_W.type())

				# model = TSception(2,(32,7680),256,9,6,128,0.2)

				if not os.path.exists(weight_dir):
					os.makedirs(weight_dir)

				if not os.path.exists(f"lightning_logs/TSception_subjects={SUBJECT_NUMBER}"):
					os.makedirs(f"lightning_logs/TSception_subjects={SUBJECT_NUMBER}")

				checkpoint_callback = ModelCheckpoint(
					monitor="val_loss",
					save_weights_only=False,
					verbose=True,
					dirpath=weight_dir,
					filename=f"Fold={fold+1}" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
				)
				logger = pl_loggers.TensorBoardLogger(
					name=f"LOTO={val_index[0]}", save_dir=f"lightning_logs/TSception_subjects={SUBJECT_NUMBER}", version = f"Fold={fold+1}"
				)

				# logger = pl_loggers.WandbLogger(project = "LGGNET", version = f"Subject-{SUBJECT_NUMBER}")

				trainer = pl.Trainer(
					gpus=NUM_GPUS,
					max_epochs=EPOCHS,
					# accumulate_grad_batches = 1,
					# auto_lr_find=True,
					callbacks=[checkpoint_callback],
					logger=logger,
					# val_check_interval=0.25,
					check_val_every_n_epoch=1,
					# precision=32,
					# resume_from_checkpoint="weights/Bi-ATransformer_sub_all_subjects/Bi-ATransformer-epoch=13-val_loss=0.64.ckpt.ckpt"
					# default_save_path = './weights'
				)

				trainer.fit(model, train_loader, test_loader)

			## outer loop
			model = LGNNet().to(DEVICE)
			checkpoint_callback_LOTO = ModelCheckpoint(
					monitor="val_loss",
					save_weights_only=False,
					verbose=True,
					dirpath=weight_dir,
					filename=f"final" + "-{epoch:02d}-{val_loss:.2f}.ckpt",
				)
			logger_LOTO = pl_loggers.TensorBoardLogger(
					name=f"LOTO={val_index[0]}", save_dir=f"lightning_logs/TSception_subjects={SUBJECT_NUMBER}", version = f"final"
				)
			weight_name, current_epoch = get_best_weight(weight_dir)
			trainer = pl.Trainer(
					gpus=NUM_GPUS,
					max_epochs=current_epoch+15,
					# accumulate_grad_batches = 1,
					# auto_lr_find=True,
					callbacks=[checkpoint_callback_LOTO],
					logger=logger_LOTO,
					# val_check_interval=0.25,
					check_val_every_n_epoch=1,
					# precision=32,
					resume_from_checkpoint=os.path.join(weight_dir, weight_name),
					# default_save_path = './weights'
				)

			train_loader = get_dataloader(X_train, y_train, BATCH_SIZE, shuffle = True)
			test_loader = get_dataloader(X_val, y_val, BATCH_SIZE, shuffle = False)

			trainer.fit(model, train_loader, test_loader)



