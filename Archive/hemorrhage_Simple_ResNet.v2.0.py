
# 1.0 ########### IMPORT LIBRARIES ###########

import numpy as np
from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import pydicom
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import RMSprop, Adam
from keras.applications import VGG19, VGG16, ResNet50

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from sklearn.utils import shuffle

import cv2
import pickle
import gc
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.ops import array_ops

# 2.0 ########### SPECIFY AND SET PARAMETERS ###########

batch_size = 32
valication_ratio = 0.1
sample_size = 2000
epochs = 3
img_size = 512

# 3.0 ########### READ IN FILES AND ORGANISE DATA STRUCTURE ###########

train_csv = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_train.csv")
train_csv['Type'] = train_csv['ID'].apply(lambda x: x.split('_')[2])
train_csv['Filename'] = train_csv['ID'].apply(lambda x: "_".join(x.split('_')[0:2]) + '.dcm')
train = train_csv[['Label', 'Filename', 'Type']].drop_duplicates().pivot(index='Filename', columns='Type', values='Label').reset_index()
print(train.head(10))
print(type(train))

train = shuffle(train)
print(train.head(10))

train_sample = train.reset_index(drop=True)  # df.reset_index(drop=True) drops the current index of the DataFrame and replaces it with an index of increasing integers. It never drops columns.
print(train_sample.head(10))

# Setting up the input images (x values or independent values)

x_head = pd.DataFrame(train_sample, columns=["Filename"])
print(x_head.head(10))
print(type(x_head))

# Setting up the correponding values for images (y values or dependent values)

y_values = pd.DataFrame(train_sample, columns=["any", "epidural", "intraventricular", "subarachnoid", "subdural"])
print(y_values.head(10))
print(type(y_values))

# 4.0 ########### CREATE A DATA GENERATOR ###########


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=3, dim=(28, 28), n_channels=3,
                 n_classes=12, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i, ] = np.load('data/' + ID + '.npy')
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# # 5.0 ########### CREATE AND INITIALISE MODEL ###########


# conv_base = ResNet50(weights='../input/models/model_weights_resnet.h5',
#                      include_top=False,
#                      input_shape=(q_size, q_size, img_channel))

# conv_base.trainable = True

# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='sigmoid'))

# # 6.0 ########### COMPILE MODEL ###########

# model.compile(optimizer=RMSprop(lr=1e-5),
#               loss='binary_crossentropy',
#               metrics=['binary_accuracy'])

# model.summary()

# # 7.0 ########### TRAIN MODEL ON DATA PASSED THROUGH THE DATA GENERATOR ###########

# history = model.fit_generator(generator=train_generator,
#                               validation_data=val_generator,
#                               epochs=epochs,
#                               class_weight=class_weight,
#                               workers=4)

# # 8.0 ########### EVALUATION: PLOT LOSS VALUES ###########

# loss = history.history['loss']
# loss_val = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'bo', label='loss_train')
# plt.plot(epochs, loss_val, 'b', label='loss_val')
# plt.title('value of the loss function')
# plt.xlabel('epochs')
# plt.ylabel('value of the loss functio')
# plt.legend()
# plt.grid()
# plt.show()

# ########### EVALUATION: PLOT ACCURACY VALUES ###########

# acc = history.history['binary_accuracy']
# acc_val = history.history['val_binary_accuracy']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, acc, 'bo', label='accuracy_train')
# plt.plot(epochs, acc_val, 'b', label='accuracy_val')
# plt.title('accuracy')
# plt.xlabel('epochs')
# plt.ylabel('value of accuracy')
# plt.legend()
# plt.grid()
# plt.show()
