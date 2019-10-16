

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

import keras
from keras.optimizers import RMSprop, Adam
from keras.applications import VGG19, VGG16, ResNet50

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from sklearn.utils import shuffle


ids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11', 'id12']
values = [342, 342, 124, 663, 435, 235, 121, 34, 768, 657, 925, 265]
names = ['train', 'validation']


labels = dict(zip(ids, values))
partition = {'train': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9'], 'validation': ['id10', 'id11', 'id12']}


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


train_Generator = DataGenerator(partition['train'], labels)


array = np.arange(len(train_Generator.list_IDs))
# print(array)
# np.random.shuffle(array)
# print(array)
# print(len(train_Generator))

print(train_Generator.batch_size)
print(train_Generator.dim)
print(train_Generator.n_channels)

x = np.empty((3))
print(x)
