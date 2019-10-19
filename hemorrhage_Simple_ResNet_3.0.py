

import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
from math import ceil, floor
import cv2
import tensorflow as tf
import keras
import sys
from keras_applications.inception_v3 import InceptionV3
from sklearn.model_selection import ShuffleSplit

test_images_dir = 'E://4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_test_images/'
train_images_dir = 'E://4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images/'

###################################################################################################################################

from math import log


def _normalize(x):
    x_max = x.max()
    x_min = x.min()
    if x_max != x_min:
        z = (x - x_min) / (x_max - x_min)
        return z
    return np.zeros(x.shape)


def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0), desired_size=(256, 256)):
    intercept, slope = img.RescaleIntercept, img.RescaleSlope
    img = img.pixel_array * slope + intercept
    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    ue = log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))

    img = _normalize(img)
    return img


def sigmoid_bsb_window(img, desired_size):
    brain_img = sigmoid_window(img, 40, 80, desired_size=desired_size)
    subdural_img = sigmoid_window(img, 80, 200, desired_size=desired_size)
    bone_img = sigmoid_window(img, 600, 2000, desired_size=desired_size)

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


dicom = pydicom.dcmread(train_images_dir + 'ID_5c8b5d701' + '.dcm')
plt.imshow(sigmoid_bsb_window(dicom, desired_size=(256, 256)))


###################################################################################################################################

def _read(path, desired_size):

    dcm = pydicom.dcmread(path)
    try:
        img = sigmoid_bsb_window(dcm, desired_size)
    except:
        img = np.zeros(desired_size)
    return img


plt.imshow(_read(train_images_dir + 'ID_5c8b5d701' + '.dcm', (128, 128)))

###################################################################################################################################


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 3),
                 img_dir=train_images_dir, *args, **kwargs):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):

        if self.labels is not None:
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))

        if self.labels is not None:
            Y = np.empty((self.batch_size, 6), dtype=np.float32)

            for i, ID in enumerate(list_IDs_temp):
                X[i, ] = _read(self.img_dir + ID + ".dcm", self.img_size)
                Y[i, ] = self.labels.loc[ID].values

            return X, Y
        else:
            for i, ID in enumerate(list_IDs_temp):
                X[i, ] = _read(self.img_dir + ID + ".dcm", self.img_size)

            return X

###################################################################################################################################


from keras import backend as K


def weighted_log_loss(y_true, y_pred):

    class_weights = np.array([2., 1., 1., 1., 1., 1.])
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.0 - eps)
    out = -(y_true * K.log(y_pred) * class_weights
            + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)
    return K.mean(out, axis=-1)


def _normalized_weighted_average(arr, weights=None):

    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1)
        return K.sum(K.dot(arr, weights), axis=1) / scl
    return K.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):

    class_weights = K.variable([2., 1., 1., 1., 1., 1.])
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * K.log(y_pred)
             + (1.0 - y_true) * K.log(1.0 - y_pred))

    loss_samples = _normalized_weighted_average(loss, class_weights)
    return K.mean(loss_samples)


def weighted_log_loss_metric(trues, preds):

    class_weights = [2., 1., 1., 1., 1., 1.]
    epsilon = 1e-7
    preds = np.clip(preds, epsilon, 1 - epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()

###################################################################################################################################


class PredictionCheckpoint(keras.callbacks.Callback):

    def __init__(self, test_df, valid_df,
                 test_images_dir=test_images_dir,
                 valid_images_dir=train_images_dir,
                 batch_size=16, input_size=(224, 224, 3)):

        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size

    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []

    def on_epoch_end(self, batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir), verbose=2)[:len(self.test_df)])


class MyDeepModel:

    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3,
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        engine = self.engine(include_top=False, weights=self.weights, input_shape=(*self.input_dims[:2], 3),
                             backend=keras.backend, layers=keras.layers,
                             models=keras.models, utils=keras.utils)

        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        self.model.compile(loss=weighted_log_loss, optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])

    def fit_and_predict(self, train_df, valid_df, test_df):
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                train_df,
                self.batch_size,
                self.input_dims,
                train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4,
            callbacks=[pred_history, scheduler]
        )
        return pred_history

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


###################################################################################################################################

def read_testset(filename="E://4.0 Projects/Hemorrhage-Detection/stage_1_sample_submission.csv"):

    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df


def read_trainset(filename="E://4.0 Projects/Hemorrhage-Detection/stage_1_train.csv"):

    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468, 312469, 312470, 312471, 312472, 312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]

    df = df.drop(index=duplicates_to_remove)
    df = df.reset_index(drop=True)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df


test_df = read_testset()
df = read_trainset()

df.head(3)
test_df.head(3)

###################################################################################################################################

ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)
train_idx, valid_idx = next(ss)

model = MyDeepModel(engine=InceptionV3, input_dims=(224, 224, 3), batch_size=16, learning_rate=5e-4,
                    num_epochs=4, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=2)

history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)

###################################################################################################################################

test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[2**i for i in range(len(history.test_predictions))])
test_df = test_df.stack().reset_index()
test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
test_df.to_csv('submission.csv', index=False)


###################################################################################################################################
