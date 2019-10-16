
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

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

train_generator = gen.flow_from_dataframe(
    train_path,
    target_size=img_size,
    shuffle=True,
    batch_size=batch_size,
)

# 5.0 ########### CREATE AND INITIALISE MODEL ###########

conv_base = ResNet50(weights='../input/models/model_weights_resnet.h5',
                     include_top=False,
                     input_shape=(q_size, q_size, img_channel))

conv_base.trainable = True

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

# 6.0 ########### COMPILE MODEL ###########

model.compile(optimizer=RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.summary()

# 7.0 ########### TRAIN MODEL ON DATA PASSED THROUGH THE DATA GENERATOR ###########

history = model.fit_generator(generator=train_generator,
                              validation_data=val_generator,
                              epochs=epochs,
                              class_weight=class_weight,
                              workers=4)

# 8.0 ########### EVALUATION: PLOT LOSS VALUES ###########

loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='loss_train')
plt.plot(epochs, loss_val, 'b', label='loss_val')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss functio')
plt.legend()
plt.grid()
plt.show()

########### EVALUATION: PLOT ACCURACY VALUES ###########

acc = history.history['binary_accuracy']
acc_val = history.history['val_binary_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, 'bo', label='accuracy_train')
plt.plot(epochs, acc_val, 'b', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()
