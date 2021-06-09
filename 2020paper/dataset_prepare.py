import os
import numpy as np
import _pickle as cPickle
import tensorflow_probability as tfp


import matplotlib.pyplot as plt
from tqdm import tqdm


deap_dir = "D:\AIproject\emotion recognition\DEAP\data_preprocessed_python"
def load_DEAP(data_dir, n_subjects = 24, only_phys = False, only_EEG = True):
    # get all files name to a list
    filenames = os.listdir(data_dir)
    filepaths = []
    for i in filenames:
        filepath = data_dir + "/" + i
        filepaths.append(filepath)
    all_data = []
    all_labels = []
    if n_subjects < 16:
        filepaths = filepaths[:n_subjects]
    else:
        filepaths = filepaths[-n_subjects:]
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

    all_data = all_data.reshape(-1, 32, all_data.shape[-1])
    all_data = all_data[:, :, 3*128:]
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    print("data shape: ", all_data.shape)

    # all_data = [all_data[:, 128*i: 128*(i+10)] for i in range(6)]
    # all_data = np.concatenate((all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5]), axis = 0)
    # # print(all_labels.shape)
    # print(all_data.shape)

    return all_data.astype(np.float16), all_labels

def feature_extraction(all_data, labels, label_type = 'valence', task = "R", C = 32, N = 10, K = 8, L = 2):
    if label_type == "valence":
            labels = labels[:, 0].squeeze()
    elif label_type == 'arousal':
        label = labels[:, 1].squeeze()
    # drop data which has label == 5
    hl_indices = np.where(labels != 5)
    labels = labels[hl_indices]
    all_data = all_data[hl_indices]
    n_samples = 8*32

    all_segmented_data = []
    for r in range(0, len(all_data), n_samples):
        print(r + n_samples, len(all_data))
        if r + n_samples < len(all_data):
            data = all_data[r: r+n_samples]
        else:
            data = all_data[r:]
        data = data[:, :, :int(data.shape[-1]/L)*L]

        # reshape: original shape: (_, C, 8064) -> (_, C, t * N * K, L)
        data = data.reshape(data.shape[0], data.shape[1], -1, L)
        # calculate single channel feature (it's the mean value)
        data = np.mean(data, axis = -1).squeeze()
        # reshape : original shape: (_, C, N*K, 1) -> (_, C, t * N, K)
        data = data.reshape(data.shape[0], data.shape[1], -1, K)
        # segmenting ovelap: overlap ratio = (N-1)/N
        segmented_data = []
        for i in range(0, data.shape[-2] - N, 1):
            segmented_data.append(np.transpose(data[:, :, i:i+N, :], (0, 2, 1, 3)))
        # reshape to calculate cov matrix
        segmented_data = np.array(segmented_data).reshape(-1, C, K)
        print(segmented_data.shape)
        # calculate pearson corvariance matrix
        segmented_corr = tfp.stats.correlation(
            segmented_data, y = None, sample_axis=-1, event_axis=-2, keepdims=False, name=None
        )
        # output = np.array(output).reshape(-1, C*C)
        all_segmented_data.append(np.array(segmented_corr))

    all_segmented_data = np.concatenate(all_segmented_data, axis = 0 )
    upper_indices = np.triu_indices_from(np.ndarray((C, C)), k=1)
    sample_matrix = np.arange(0, C*C).reshape((C, C))
    upper_values  = list(sample_matrix[upper_indices])
    all_segmented_data = all_segmented_data.reshape(-1, N, C*C)
    output = all_segmented_data[:, :, upper_values]
    print(output.shape)
    # output = []

    # for i in range(len(all_segmented_data)):
    #     # get only the upper triangular of the covariance matrix)
    #     cov_matrix = all_segmented_data[i]
    #     output.append(list(cov_matrix[upper_indices]))

    # output = np.array(output).reshape(int(len(all_segmented_data)/N), N, -1)

    if task == 'C':
        labels = np.where(labels > 5, 1, 0)
    segmented_labels = np.repeat(labels, repeats = int(output.shape[0]/len(labels)), axis = 0)
    print(segmented_labels)
    return output, np.array(segmented_labels)


if __name__ == "__main__":
    data, labels = load_DEAP(deap_dir, n_subjects = 1)
    output, out_labels = feature_extraction(data, labels)
    print(output.shape)
    print(out_labels)
