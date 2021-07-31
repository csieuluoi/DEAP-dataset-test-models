import torch
import numpy as np
from custom_dataset import DEAP_dataset
from custom_transforms import CustomTransform
from preprocessing import dataset_prepare
import matplotlib.pyplot as plt
from model import ACTransformer, DEVICE
# TRANSFORM = CustomTransform(
#     scale = None, 
#     n_scale = 32, 
#     wavelet = "morl", 
#     sampling_rate = 128, 
#     normalized=True
# )

# if __name__ =='__main__':
#     X_train, y_train, X_test, y_test = dataset_prepare(
#         segment_duration = 3, 
#         load_all = True, 
#         return_dataset = False, 
#         sampling_rate = 128
#     )
    
#     print("train x shape:", X_train.shape)
#     print("train y shape:", y_train.shape)

#     dataset = DEAP_dataset(data = X_train, labels = y_train, transform = TRANSFORM)

#     for d in dataset:
#         print(d)
#         print(d[0].shape)
#         plt.figure()
#         plt.imshow(d[0][0, 0])
#         plt.show()
#         break


if __name__ == "__main__":
    feature_dim = 80 # this is # output features from CNN 
    num_classes = 2
    dim = 128
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
        n_steps, eeg_channels, 
        cwa_hidden,
        cnn_in_channels, 
        dim_head = 64, 
        pool = "cls", 
        dropout = dropout, 
        emb_dropout = emb_dropout,
    ).to(DEVICE)


    input = torch.randn(2, 3, 32, 32, 128).to(DEVICE)

    output = model(input)
    print(output.shape)
