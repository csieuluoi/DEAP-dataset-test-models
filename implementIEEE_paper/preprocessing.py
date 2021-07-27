import numpy as np
from utils import load_DEAP
from sklearn.model_selection  import train_test_split
from torch.utils.data import TensorDataset
import torch
from sklearn.utils import shuffle
import os

DATA_DIR = "D:\AIproject\emotion recognition\DEAP\data_preprocessed_python"



def baseline_removal(data):
    """ 
    calculate the baseline signal per second 
    then subtract that baseline from the signal
    """
    # duration of the baseline
    baseline_dur = 3 
    # signal's sampling rate
    sampling_rate = 128
    preprocessed = []
    # loop through the data array (n_instance, n_channels, n_samples)
    for ins in range(data.shape[0]):
        preprocessed_ins = []
        for c in range(data.shape[1]):
            signal = data[ins, c]
            # get all 3 second baseline segment and split in to 3 1-second segments
            all_baseline = np.split(signal[:baseline_dur*sampling_rate], 3)
            signal = signal[baseline_dur*sampling_rate:]
            # calculate the per second mean baseline
            baseline_per_second = np.mean(all_baseline, axis = 0)
            # print(baseline_per_second.shape)
            baseline_to_remove = np.tile(baseline_per_second, int(len(signal)/sampling_rate))
            signal_baseline_removed = signal - baseline_to_remove
    
            signal_split = signal_baseline_removed.reshape(-1, 3*128)
            
            preprocessed_ins.append(signal_split)
        
        preprocessed.append(preprocessed_ins)
        

    return np.array(preprocessed).transpose(0, 2, 1, 3)


def dataset_prepare(segment_duration = 3, n_subjects = 1, load_all = False, single_subject = False, sampling_rate = 128):
    data = load_DEAP(DATA_DIR, n_subjects = n_subjects, single_subject=single_subject,load_all = load_all)    
    if load_all or single_subject:
        s1, s1_labels, s1_names = data
        s1_labels = np.repeat(s1_labels.reshape(-1, 1), 20)

        s1_preprocessed = baseline_removal(s1)
        b, s, c, n = s1_preprocessed.shape
        s1_preprocessed = s1_preprocessed.reshape(b*s, c, segment_duration, sampling_rate).transpose(0, 2, 1, 3)
        print("preprocesed data shape: ", s1_preprocessed.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(s1_preprocessed, s1_labels, test_size = 0.2, stratify = s1_labels, shuffle = True, random_state = 29)
    else:
        train_data, train_labels, train_names, test_data, test_labels, test_names = data
        y_train = np.repeat(train_labels.reshape(-1, 1), 20)
        X_train = baseline_removal(train_data)
        b, s, c, n = X_train.shape
        X_train = X_train.reshape(b*s, c, segment_duration, sampling_rate).transpose(0, 2, 1, 3)

        y_test = np.repeat(test_labels.reshape(-1, 1), 20)
        X_test = baseline_removal(test_data)
        b, s, c, n = X_test.shape
        X_test = X_test.reshape(b*s, c, segment_duration, sampling_rate).transpose(0, 2, 1, 3)

    train_x = torch.Tensor(X_train) # transform to torch tensor
    train_y = torch.Tensor(y_train)
    test_x = torch.Tensor(X_test) # transform to torch tensor
    test_y = torch.Tensor(y_test)

    train_dataset = TensorDataset(train_x, train_y.long()) # create your datset
    test_dataset = TensorDataset(test_x, test_y.long())

    return train_dataset, test_dataset


def dataset_prepare_for_KF(n_subjects = 1):
    s1, s1_labels, s1_names = load_DEAP(DATA_DIR, n_subjects = n_subjects, single_subject=True)    
    s1_labels = np.repeat(s1_labels.reshape(-1, 1), 20)

    s1_preprocessed = baseline_removal(s1)
    s1_preprocessed = s1_preprocessed.reshape(800, 32, 3, 128).transpose(0, 2, 1, 3)

    return s1_preprocessed, s1_labels
    
if __name__ == "__main__":
    train_dataset, test_dataset =  dataset_prepare()

