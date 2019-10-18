
import numpy as np
import pandas as pd
import random

import os

import pydicom as dicom
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop, Adam
from keras.applications import VGG19, VGG16, ResNet50

# import warnings
# warnings.filterwarnings("ignore")

path_base = 'E://4.0 Projects/Hemorrhage-Detection/'
path_train_img = 'Hemorrhage-Detection-Data/stage_1_train_images/'
path_test_img = 'Hemorrhage-Detection-Data/stage_1_test_images/'

list_train_img = os.listdir(path_train_img)
list_test_img = os.listdir(path_test_img)

train_df = pd.read_csv(path_base + 'stage_1_train.csv')
submission_df = pd.read_csv(path_base + 'stage_1_sample_submission.csv')

q_size = 200
img_channel = 3
num_classes = 6

train_df['sub_type'] = train_df['ID'].str.split("_", n=3, expand=True)[2]
train_df['PatientID'] = train_df['ID'].str.split("_", n=3, expand=True)[1]

submission_df['sub_type'] = submission_df['ID'].str.split("_", n=3, expand=True)[2]
submission_df['PatientID'] = submission_df['ID'].str.split("_", n=3, expand=True)[1]

print(train_df.head(10))
print(submission_df.head(10))

train_data_pivot = train_df[['Label', 'PatientID', 'sub_type']].drop_duplicates().pivot(index='PatientID', columns='sub_type', values='Label')
test_data_pivot = submission_df[['Label', 'PatientID', 'sub_type']].drop_duplicates().pivot(index='PatientID', columns='sub_type', values='Label')

# print(train_data_pivot.head(10))
# print(test_data_pivot.head(10))

percentage = 0.2
num_train_img = int(percentage * len(train_data_pivot.index))
num_test_img = len(test_data_pivot.index)

# print('num_train_data:', len(list_train_img), num_train_img)
# print('num_test_data:', len(list_test_img))

list_train_img = list(train_data_pivot.index)
list_test_img = list(test_data_pivot.index)

random_train_img = random.sample(list_train_img, num_train_img)
y_train_org = train_data_pivot.loc[random_train_img]

y_train, y_val = train_test_split(y_train_org, test_size=0.3)
y_test = test_data_pivot


print(type(y_train))
print(type(y_val))

print(y_test.head(10))

class_weight = dict(zip(range(0, num_classes), y_train.sum() / y_train.sum().sum()))
print(class_weight)

print(range(0, num_classes), y_train.sum() / y_train.sum().sum())
print(y_train.sum().sum())
print(y_train.sum())
print(y_train.head(10))
print(y_train.sum() / y_train.sum().sum())


class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, labels, batch_size, img_size, img_channel, num_classes, shuffle=True):
        self.path = path
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_channel = img_channel
        self.num_classes = num_classes
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

    def rescale_pixelarray(self, dataset):
        image = dataset.pixel_array
        rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
        rescaled_image[rescaled_image < -1024] = -1024
        return rescaled_image

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.img_size, self.img_size))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            data_file = dicom.dcmread(self.path + '/ID_' + ID + '.dcm')
            img = self.rescale_pixelarray(data_file)
            img = cv2.resize(img, (self.img_size, self.img_size))
            X[i, ] = img
            y[i, ] = self.labels.loc[ID]
        X = np.repeat(X[..., np.newaxis], 3, -1)
        X = X.astype('float32')
        X -= X.mean(axis=0)
        std = X.std(axis=0)
        X /= X.std(axis=0)
        return X, y


conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(q_size, q_size, img_channel))
conv_base.trainable = True

batch_size = 32

train_generator = DataGenerator(path_train_img, list(y_train.index), y_train, batch_size, q_size, img_channel, num_classes)
val_generator = DataGenerator(path_train_img, list(y_val.index), y_val, batch_size, q_size, img_channel, num_classes)

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['binary_accuracy'])

model.summary()

epochs = 1
history = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weight, workers=4)

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
