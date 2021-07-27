from Fnet import FNet
import torch
import numpy as np


# d_model = 640
# fnet = FNet(d_model=d_model, expansion_factor=2, dropout=0.5, num_layers=6, n_classes = 1)

if __name__ =='__main__':
    # input = torch.randn((3, 32, d_model))
    # out = fnet(input)
    # print(out.size())
    # random = np.random.randint(1, 10, 1280)
    # print(random)
    # low_index = np.asarray((random < 5) & (random >1)).nonzero()[0]
    # high_index = np.asarray((random > 5) & (random >1)).nonzero()[0]

    # print(low_index)
    # print()
    # keep_index = np.concatenate((low_index, high_index))
    # print(keep_index)


    arr = np.arange(0, 48).reshape(2, 4, 6)

    mean = np.mean(arr, axis = 2)
    std = np.std(arr, axis = 2)
=
    arr = (arr - mean[:, :,np.newaxis]) / std[:, : ,np.newaxis]
    print(arr)
