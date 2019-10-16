
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

# ########### CREATE A DATA GENERATOR ###########

# gen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
# )

# train_generator = gen.flow_from_dataframe(
#     train_path,
#     target_size=img_size,
#     shuffle=True,
#     batch_size=batch_size,
# )

# ########### Design model ###########

# model = model()

# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='sigmoid'))


# ########### Compile model ###########

# model.compile(optimizer = RMSprop(lr=1e-5),
#               loss='binary_crossentropy',
#               metrics=['binary_accuracy'])
# model.summary()


# ########### Train model on dataset ###########

# history = model.fit_generator(
#     train_generator,
#     validation_data=valid_generator,
#     epochs=epochs,
#     steps_per_epoch=len(image_files) // batch_size,
#     validation_steps=len(valid_image_files) // batch_size,
# )
