from utils import *

import scipy
import numpy as np

from scipy.signal import butter, lfilter


from tqdm import tqdm

eeg_bands = {"theta": [4, 7], "alpha": [8, 15], "beta": [16, 31], "gamma": [32, 45]}


def butter_bandpass(lowcut, highcut, fs=128, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs=128, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_DE_EEG_bands(x):
    theta = butter_bandpass_filter(x, eeg_bands["theta"][0], eeg_bands["theta"][1])
    alpha = butter_bandpass_filter(x, eeg_bands["alpha"][0], eeg_bands["alpha"][1])
    beta = butter_bandpass_filter(x, eeg_bands["beta"][0], eeg_bands["beta"][1])
    gamma = butter_bandpass_filter(x, eeg_bands["gamma"][0], eeg_bands["gamma"][1])

    de_features = []

    for signal in [theta, alpha, beta, gamma]:
        de_features.append(calculate_DE(signal))

    return de_features


def calculate_DE(x):
    de = 1 / 2 * np.log2(2 * np.pi * np.e * np.std(x))

    return de


def extract_one_subject_features(
    sub_idx=1,
    feature_type_func=extract_DE_EEG_bands,
    num_classes=2,
    save_dir="data",
    normalize_signal=False,
    segmenting=False,
    segment_duration=3,
):
    data, y_val, y_ar = get_one_subject(sub_idx, num_classes=num_classes)
    """====================================normalizing signals========================================"""
    m = np.mean(data[:, :, : 128 * 3])
    std = np.std(data[:, :, : 128 * 3])

    if normalize_signal:
        data = (data[:, :, 128 * 33 :] - m) / std
    """====================================end normalizing============================================"""
    """====================================segmenting================================================="""
    if segmenting:
        b, c, n = data.shape
        data = data.reshape(-1, c, segment_duration * 128)
        y_val = np.repeat(y_val.reshape(-1, 1), int(n / (segment_duration * 128)))
        y_ar = np.repeat(y_ar.reshape(-1, 1), int(n / (segment_duration * 128)))
    """====================================end segmenting============================================="""
    sub_features = []
    sub_ar_labels = []
    sub_val_labels = []
    for i in range(data.shape[0]):
        ins_features = []
        for c in range(data.shape[1]):
            x = data[i, c]
            # if normalize_signal:
            #     x = (x - np.mean(x)) / np.std(x)
            c_features = feature_type_func(x=x)
            ins_features.append(c_features)
        sub_features.append(ins_features)
    sub_val_labels.append(y_val)
    sub_ar_labels.append(y_ar)

    sub_features = np.array(sub_features)
    sub_val_labels = np.array(sub_val_labels)
    sub_ar_labels = np.array(sub_ar_labels)
    print(sub_features)
    print(sub_val_labels)
    print(sub_ar_labels)

    print(sub_features.shape)
    print(sub_val_labels.shape)
    print(sub_ar_labels.shape)

    sub_val_labels = sub_val_labels.reshape(-1, 1)
    sub_ar_labels = sub_ar_labels.reshape(-1, 1)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.save(f"data/sub_{sub_idx}_de_features.npy", sub_features)
    np.save(f"data/sub_{sub_idx}_val{num_classes}_features.npy", sub_val_labels)
    np.save(f"data/sub_{sub_idx}_ar{num_classes}_features.npy", sub_ar_labels)

    return sub_features, sub_val_labels, sub_ar_labels


def extract_all_features(
    feature_type_func=extract_DE_EEG_bands,
    n_subject=33,
    num_classes=2,
    save_dir="data",
    normalize_signal=False,
):
    all_features = []
    all_val_labels = []
    all_ar_labels = []
    for sub_idx in tqdm(range(1, n_subject)):
        data, y_val, y_ar = get_one_subject(sub_idx, num_classes=num_classes)

        for i in range(data.shape[0]):
            ins_features = []
            for c in range(data.shape[1]):
                if normalize_signal:
                    x = data[i, c]
                    x = (x - np.mean(x)) / np.std(x)
                c_features = feature_type_func(x=x)
                ins_features.append(c_features)
            all_features.append(ins_features)
        all_val_labels.append(y_val)
        all_ar_labels.append(y_ar)

    all_features = np.array(all_features)
    all_val_labels = np.array(all_val_labels)
    all_ar_labels = np.array(all_ar_labels)

    all_val_labels = all_val_labels.reshape(-1, 1)
    all_ar_labels = all_ar_labels.reshape(-1, 1)

    print(all_features)
    print(all_val_labels)
    print(all_ar_labels)

    print(all_features.shape)
    print(all_val_labels.shape)
    print(all_ar_labels.shape)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.save("data/all_de_features.npy", all_features)
    np.save(f"data/all_val{num_classes}_features.npy", all_val_labels)
    np.save(f"data/all_ar{num_classes}_features.npy", all_ar_labels)

    return all_features, all_val_labels, all_ar_labels


if __name__ == "__main__":
    extract_all_features(
        feature_type_func=extract_DE_EEG_bands, n_subject=33, num_classes=2
    )
