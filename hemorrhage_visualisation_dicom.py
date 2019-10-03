
from glob import glob
import os
from os import listdir
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
from PIL import Image
import seaborn as sns
from random import randrange

import pydicom

# Specifiy directories
train_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images/"
test_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data-Data/stage_1_test_images/"

# Reading in CSV files
train_df = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_train.csv")
sample_df = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_sample_submission.csv")

print(os.listdir())

train = sorted(glob("E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images/*.dcm"))
test = sorted(glob("E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_test_images/*.dcm"))

print(len(train))
print(len(test))

train_files = os.listdir(train_dir)
print(train_files[0:10])

dataset = pydicom.dcmread(train_dir + "ID_c5c23af94.dcm")
print(dataset)

image = dataset.pixel_array.flatten()
print(image)

# Showing the distribution of pixes in 10 images

fig, ax = plt.subplots(2, 1, figsize=(20, 10))
for file in train_files[0:10]:
    dataset = pydicom.dcmread(train_dir + file)
    image = dataset.pixel_array.flatten()
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    sns.distplot(image.flatten(), ax=ax[0])
    sns.distplot(rescaled_image.flatten(), ax=ax[1])
ax[0].set_title("Raw pixel array distributions for 10 examples")
ax[1].set_title("HU unit distributions for 10 examples")
plt.show()

fig = plt.figure(figsize=(15, 10))
columns = 5
rows = 4

for i in range(1, columns * rows + 1):
    dataset = pydicom.dcmread(train_dir + train_files[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.show()

##### Load Data ####


def window_image(img, window_center, window_width, intercept, slope):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def get_first_of_dicom_field_as_int(x):
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


case = 5
data = pydicom.dcmread(train[case])

window_center, window_width, intercept, slope = get_windowing(data)

# displaying the image
img = pydicom.read_file(train[case]).pixel_array
img = window_image(img, window_center, window_width, intercept, slope)
plt.imshow(img, cmap=plt.cm.bone)
plt.grid(False)
plt.show()
print(data)


def view_images(images, title='', aug=None):

    width = 5
    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15, 5))

    for im in range(0, height * width):

        data = pydicom.read_file(os.path.join(train_dir, images[im] + '.dcm'))
        image = data.pixel_array
        window_center, window_width, intercept, slope = get_windowing(data)

        image_windowed = window_image(image, window_center, window_width, intercept, slope)

        i = im // width  # 8/5
        j = im % width
        axs[i, j].imshow(image_windowed, cmap=plt.cm.bone)
        axs[i, j].axis('off')

    plt.suptitle(title)
    plt.show()


train_df['image'] = train_df["ID"].apply(lambda x: "_".join(x.split("_")[0:2]))
train_df['diagnosis'] = train_df["ID"].apply(lambda x: x.split("_")[2])

view_images(train_df[(train_df['diagnosis'] == 'epidural') & (train_df['Label'] == 1)][:10].image.values, title='Images with epidural')
view_images(train_df[(train_df['diagnosis'] == 'intraparenchymal') & (train_df['Label'] == 1)][:10].image.values, title='Images with intraparenchymal')
view_images(train_df[(train_df['diagnosis'] == 'intraventricular') & (train_df['Label'] == 1)][:10].image.values, title='Images with intraventricular')
view_images(train_df[(train_df['diagnosis'] == 'subarachnoid') & (train_df['Label'] == 1)][:10].image.values, title='Images with subarachnoid')
view_images(train_df[(train_df['diagnosis'] == 'subdural') & (train_df['Label'] == 1)][:10].image.values, title='Images with subarachnoid')
