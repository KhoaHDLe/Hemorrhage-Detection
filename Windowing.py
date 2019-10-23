

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import os


test_images_dir = 'E://4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_test_images/'
train_images_dir = 'E://4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images/'

BASE_PATH = 'E://4.0 Projects/Hemorrhage-Detection/'
TRAIN_DIR = 'Hemorrhage-Detection-Data/stage_1_train_images/'
TEST_DIR = 'Hemorrhage-Detection-Data/stage_1_test_images/'

train_df = pd.read_csv(BASE_PATH + 'stage_1_train.csv')
sub_df = pd.read_csv(BASE_PATH + 'stage_1_sample_submission.csv')
print(train_df.head(10))

train_df['filename'] = train_df['ID'].apply(lambda x: 'ID_' + x.split('_')[1] + ".dcm")
train_df['type'] = train_df['ID'].apply(lambda x: x.split('_')[2])
print(train_df.head(10))

train = train_df[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label')
print(train.head(5))

train = train_df[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()
print(train.head(5))

hem_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


def load_random_images():
    image_names = [list(train[train[h_type] == 1].sample(1)['filename'])[0] for h_type in hem_types]
    image_names += list(train[train['any'] == 0].sample(5)['filename'])

    return [pydicom.read_file(os.path.join(train_images_dir, img_name)) for img_name in image_names]


def view_images(images):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15, 5))

    for im in range(0, height * width):
        image = images[im]
        i = im // width
        j = im % width
        axs[i, j].imshow(image, cmap=plt.cm.bone)
        axs[i, j].axis('off')
        title = hem_types[im] if im < len(hem_types) else 'normal'
        axs[i, j].set_title(title)

    plt.show()


imgs = load_random_images()
view_images([img.pixel_array for img in imgs])
