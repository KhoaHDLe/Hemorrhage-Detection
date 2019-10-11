
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

batch_size = 32
valication_ratio = 0.1
sample_size = 2000
epochs = 3
img_size = 512

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


########### CREATE A DATA GENERATOR ###########


########### Design model ###########


########### Train model on dataset ###########
