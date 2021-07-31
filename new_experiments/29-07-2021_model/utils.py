import os
import torch
import numpy as np
import pandas as pd
import _pickle as cPickle
import random
from sklearn.utils import shuffle
import pywt


def load_DEAP(data_dir, n_subjects = 26, single_subject = False, load_all = False, only_phys = False, only_EEG = True, label_type = [0, 2]):
    ## label_type [arousal: 0, valence: 1, n_classes]
    # get all files name to a list
    filenames = os.listdir(data_dir)
    filepaths = []
    for i in filenames:
        filepath = data_dir + "/" + i
        filepaths.append(filepath)

    if single_subject:

        train_paths = [filepaths[n_subjects-1]]
        train_names = [filenames[n_subjects-1]]
        # print(train_paths, "\n", train_names)
        train_data, train_labels = load_with_path(train_paths, label_type = label_type)

        return train_data, train_labels, train_names

    if load_all:

        train_paths = filepaths
        train_names = filenames
        # print(train_paths, "\n", train_names)
        train_data, train_labels = load_with_path(train_paths, label_type = label_type)

        return train_data, train_labels, train_names

    filepaths, filenames = shuffle(filepaths, filenames, random_state = 29)

    train_paths = filepaths[:n_subjects]
    test_paths = filepaths[n_subjects:]
    train_names = filenames[:n_subjects]
    test_names = filenames[n_subjects:]
    print(train_names)
    print(test_names)
    train_data, train_labels = load_with_path(train_paths, label_type = label_type)
    test_data, test_labels = load_with_path(test_paths, label_type = label_type)

        # keep_index = np.concatenate((low_index, high_index, mid_index))
        # all_labels = all_labels[keep_index].astype(np.uint8)
        # all_data = all_data[keep_index]
    print("train shape: ", train_data.shape)
    print("test shape: ", test_data.shape)

    return train_data, train_labels, train_names, test_data, test_labels, test_names


def load_with_path(filepaths, label_type = [0, 2], only_phys = False, only_EEG = True):
    all_data = []
    all_labels = []

    print(len(filepaths))
    for filepath in filepaths:
        loaddata = cPickle.load(open(filepath, 'rb'), encoding="latin1",)
        labels = loaddata['labels']
        new_data = loaddata['data'].astype(np.float32)
        if only_phys:
            new_data = new_data[:, 32:, :]
        elif only_EEG:
            new_data = new_data[:, :32, :]
        all_labels.append(labels)
        all_data.append(new_data)
    all_labels = np.array(all_labels)

    all_data = np.array(all_data)

    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    all_data = all_data.reshape(-1, all_data.shape[-2], all_data.shape[-1])
    print("data shape:", all_data.shape)

    # print("all label before categorizing", all_labels)

    if label_type[1] == 1:
        pass
    elif label_type[1] == 2:
        # low_index = np.asarray(all_labels < 5).nonzero()[0]
        # high_index = np.asarray(all_labels > 5).nonzero()[0]

        # keep_index = np.concatenate((low_index, high_index))
        # all_labels = all_labels[keep_index]
        # all_labels = np.where(all_labels > 5, 1, 0)
        # all_data = all_data[keep_index]
        all_labels = labels_quantization(all_labels, 2)
    else:
        low_index = np.asarray(all_labels < 4).nonzero()[0]
        high_index = np.asarray(all_labels > 6).nonzero()[0]
        mid_index = np.asarray((4 <= all_labels) & (all_labels >= 6)).nonzero()
        all_labels[low_index] = 0
        all_labels[high_index] = 2
        all_labels[mid_index] = 1
    all_labels = all_labels[label_type[0]].squeeze()

    # print("all label after categorizing", all_labels)

    return all_data, all_labels


def labels_quantization(labels, num_classes):
    new_labels = labels
    if num_classes == 2:

        median_val = 5
        median_arousal = 5

        labels_val = np.zeros(new_labels.shape[0])
        labels_arousal = np.zeros(new_labels.shape[0])

        labels_val[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= median_val)] = 0
        labels_val[(median_val < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 1

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= median_arousal)] = 0
        labels_arousal[(median_arousal < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 1

    elif num_classes == 3:
        low_value = 4
        high_value = 6

        labels_val = np.zeros(new_labels.shape[0])
        labels_arousal = np.zeros(new_labels.shape[0])

        labels_val[(1 <= new_labels[:, 0]) & (new_labels[:, 0] <= low_value)] = 0
        labels_val[(low_value < new_labels[:, 0]) & (new_labels[:, 0] <= high_value)] = 1
        labels_val[(high_value < new_labels[:, 0]) & (new_labels[:, 0] <= 9)] = 2

        labels_arousal[(1 <= new_labels[:, 1]) & (new_labels[:, 1] <= low_value)] = 0
        labels_arousal[(low_value < new_labels[:, 1]) & (new_labels[:, 1] <= high_value)] = 1
        labels_arousal[(high_value < new_labels[:, 1]) & (new_labels[:, 1] <= 9)] = 2

    output_labels = [labels_val, labels_arousal]

    return np.array(output_labels)


def generate_scalogram(data, scale = None, n_scale = 32, wavelet = 'morl', sampling_rate = 128, normalized = True):
    ## data shape : (n_steps, N_channel, N_samples) -> eg: ( 10, 32, 8064)
    ## segment_length: length of the segment in second
    # print(data.shape)
    if scale == None:
        scale = np.geomspace(2.4, 26, num = n_scale)
    """ scales and wavelet name to use:
    # scale = np.geomspace(0.71, 8, num=32)
    # wavelet_name = 'mexh'

    # scale = np.geomspace(2.31, 26, num=32)
    # wavelet_name = 'morl'

    # scale = np.geomspace(0.8533, 9.6, num=32)
    # wavelet_name = 'cgau1'

    # scale = np.geomspace(1.13, 12.8, num=32)
    # wavelet_name = 'cgau2'

    scale = np.geomspace(1.42, 16, num=32)
    wavelet_name = 'cgau4'

    # scale = np.geomspace(1.42, 16, num=32)
    # wavelet_name = 'cmor'

    # scale = np.geomspace(1.42, 16, num=32)
    # wavelet_name = 'cmor0.75-0.5'

    # scale = np.geomspace(0.56, 6.4, num=32)
    # wavelet_name = 'gaus1'

    # scale = np.geomspace(1.42, 16, num=32)
    # wavelet_name = 'gaus4'

    # scale = np.geomspace(1.42, 16, num=32)
    # wavelet_name = 'fbsp'
    """
    rs = False
    # print("generating scalogram datashape:", data.shape)

    if len(data.shape)==3:
        rs = True
        steps, channels, samples = data.shape
        data = data.reshape(steps*channels, samples)
    try:
        coefs, _ = pywt.cwt(data, scale, wavelet = wavelet, sampling_period = 1/sampling_rate, axis = -1) ## (32, _, 128)
        # print("shape of wavelet coefs before transposing", coefs.shape)

    except:
        print("--------some bug happen hear!!!------------------------------------------------------------------")
        print(data.shape)

    if len(coefs.shape) == 3:
        coefs = np.transpose(coefs, (1, 0, 2))  ## (_, 32, 128)
    # print("shape of wavelet coefs after transposing", coefs.shape)
    if normalized:
        if len(coefs) > 1:
            sc = []
            for coef in coefs:
                energy = abs(coef*coef)
                sc.append(100*energy/(np.sum(energy)))
                # sc.append(np.log10(energy - np.mean(energy)))
                # sc.append(librosa.core.power_to_db(energy)) #S_db ~= 10 * log10(S) - 10 * log10(ref)
        else:
            energy = abs(coefs*coefs)
            sc = 100*energy/(np.sum(energy))
            # sc = np.log10(energy - np.mean(energy))
            # sc = librosa.core.power_to_db(energy)

    else:
        sc = abs(coefs*coefs)

    sc = np.array(sc)
    if rs:
        sc = sc.reshape(steps, channels, -1, samples)

    return sc

# def from_fig_to_image(imgs, resize = False, new_size = (112, 112), grayscale = False):
#     output = []
#     for im_arr in imgs:
#         ## create cmap figure from 1d array and return rgb image

#         fig = plt.figure(frameon=False, tight_layout = (0, 0, 0, 0), figsize=(2, 4))
#         plt.imshow(im_arr, cmap = 'jet', vmin=0.0, vmax=1.0)
#         plt.axis('off')
#         fig.canvas.draw()
#         # convert canvas to image
#         img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#         img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         # plt.show()

#         plt.close()

#         # img is rgb, convert to opencv's default bgr
#         # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         # to gray
#     #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         indices = np.where(img[:, :,  0] != 255)

#         img = img[indices[0][0]: indices[0][-1], indices[1][0]: indices[1][-1]]

#         if grayscale:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         if resize:
#             img = cv2.resize(img, new_size)
#         output.append(img)

#     return np.array(output)


