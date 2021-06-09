import tensorflow_datasets as tfds

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Input
from tensorflow.keras import Sequential, Model
from dataset_prepare import feature_extraction
from sklearn.model_selection import train_test_split
from dataset_prepare import load_DEAP, feature_extraction
import matplotlib.pyplot as plt
deap_dir = "D:\AIproject\emotion recognition\DEAP\data_preprocessed_python"

class Model2020Object(Model):
  def __init__(self):
    super(Model2020Object, self).__init__()
    self.lstm1 = LSTM(units = 256, activation='relu')
    self.lstm1 = LSTM(units = 256, activation='relu', return_sequences = True)
    self.dropout = Dropout(0.5)
    self.d1 = Dense(1, activation='sigmoid')

  def call(self, x):
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.dropout(x)
    x = self.d1(x)
    return x

def Model2020(input_shape, task = 'R'):
    model = Sequential()
    model.add(LSTM(units = 256, activation = 'relu', input_shape = (10, 496), return_sequences = True, kernel_initializer = tf.keras.initializers.VarianceScaling()))
    model.add(LSTM(units = 256, activation = 'relu', return_sequences = False, kernel_initializer = tf.keras.initializers.VarianceScaling()))
    model.add(Dropout(0.5))
    if task == "C":
        activation_function = "sigmoid"
    else:
        activation_function = "linear"
    model.add(Dense(1, activation = activation_function, kernel_initializer = tf.keras.initializers.GlorotNormal()))

    return model


if __name__ == '__main__':
    # data = np.random.rand(14, 10, 496)
    # y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    task = "C"
    data, labels = load_DEAP(deap_dir, n_subjects = 10)
    output, out_labels = feature_extraction(data, labels, label_type = 'valence', task = task, L = 15)
    del data
    del labels
    # print(output.shape, out_labels.shape)
    # X_train, X_test, y_train, y_test = train_test_split(output, out_labels, stratify = out_labels, test_size = 0.1, shuffle = True, random_state = 42)
    # # y = tf.keras.utils.to_categorical(y)
    # print(X_train.shape, y_train.shape)

    model = Model2020((10, 496), task)
    # opt = tf.keras.optimizers.RMSprop(learning_rate = 1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    if task == "R":
        loss = "mse"
        metrics = ["mse"]
    else:
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    model.compile(loss = loss, optimizer = opt, metrics = metrics)
    history = model.fit(output, out_labels,
        epochs = 50,
        batch_size = 240,
        validation_split = 0.1,
        shuffle = True)

    if task == "C":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for losses
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    model.save("first_model.h5")

