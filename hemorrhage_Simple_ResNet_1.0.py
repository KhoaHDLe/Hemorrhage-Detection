
import os
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

from keras import layers

from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.ops import array_ops
from tqdm import tqdm

from keras import backend as K
import tensorflow as tf

BASE_PATH = 'E://4.0 Projects/Hemorrhage-Detection/'
TRAIN_DIR = 'Hemorrhage-Detection-Data/stage_1_train_images/'
TEST_DIR = 'Hemorrhage-Detection-Data/stage_1_test_images/'

train_df = pd.read_csv(BASE_PATH + 'stage_1_train.csv')
sub_df = pd.read_csv(BASE_PATH + 'stage_1_sample_submission.csv')

train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])
sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])

print(train_df.shape)
print(train_df.head())

test_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
print(test_df.shape)
print(test_df.head())

np.random.seed(1749)
sample_files = np.random.choice(os.listdir(BASE_PATH + TRAIN_DIR), 1000)
sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]

pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()
print(pivot_df.shape)
print(pivot_df.head())


def window_image(img, window_center, window_width, intercept, slope, rescale=True):

    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)

    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def save_and_resize(filenames, load_dir):
    save_dir = 'E://4.0 Projects/Hemorrhage-Detection/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')

        dcm = pydicom.dcmread(path)
        window_center, window_width, intercept, slope = get_windowing(dcm)
        img = dcm.pixel_array
        img = window_image(img, window_center, window_width, intercept, slope)

        resized = cv2.resize(img, (224, 224))
        res = cv2.imwrite(new_path, resized)

        if not res:
            print('Failed')


# save_and_resize(filenames=sample_files, load_dir=BASE_PATH + TRAIN_DIR)
# save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)

BATCH_SIZE = 16


def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.1,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images,
        validation_split=0.2
    )


def create_test_gen():
    return ImageDataGenerator().flow_from_dataframe(
        test_df,
        directory='E://4.0 Projects/Hemorrhage-Detection/save_and_resize',
        x_col='filename',
        class_mode=None,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=False
    )


def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        pivot_df,
        directory='E://4.0 Projects/Hemorrhage-Detection/save_and_resize',
        x_col='filename',
        y_col=['any', 'epidural', 'intraparenchymal',
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='other',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        subset=subset
    )


# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, 'training')
val_gen = create_flow(data_generator, 'validation')
test_gen = create_test_gen()

densenet = DenseNet121(
    weights='E://4.0 Projects/Hemorrhage-Detection/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224, 224, 3)
)


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(6, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )

    return model


model = build_model()
model.summary()

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

total_steps = sample_files.shape[0] / BATCH_SIZE

history = model.fit_generator(
    train_gen,
    steps_per_epoch=total_steps * 0.80,
    validation_data=val_gen,
    validation_steps=total_steps * 0.2,
    callbacks=[checkpoint],
    epochs=11
)

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

model.load_weights('model.h5')
y_test = model.predict_generator(
    test_gen,
    steps=len(test_gen),
    verbose=1
)

# test_df[['ID', 'Label']].head(10)