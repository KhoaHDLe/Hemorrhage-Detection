
# 1.0 ########### IMPORT LIBRARIES ###########

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

# 2.0 ########### SPECIFY AND SET PARAMETERS ###########


# 3.0 ########### READ IN FILES AND ORGANISE DATA STRUCTURE ###########

base_path = ('E://4.0 Projects/Hemorrhage-Detection/')
train_dir = ('Hemorrhage-Detection-Data/stage_1_train_images/')
test_dir = ('Hemorrhage-Detection-Data/stage_1_test_images/')

train_df = pd.read_csv(base_path + 'stage_1_train.csv')
sub_df = pd.read_csv(base_path + 'stage_1_sample_submission.csv')


train_df['filename'] = train_df['ID'].apply(lambda x: "_".join(x.split('_')[0:2]) + '.png')
train_df['type'] = train_df['ID'].apply(lambda x: x.split('_')[2])

sub_df['filename'] = sub_df['ID'].apply(lambda x: "_".join(x.split('_')[0:2]) + '.png')
sub_df['type'] = sub_df['ID'].apply(lambda x: x.split('_')[2])

submission_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
print(submission_df.shape)
print(submission_df.head(10))


print(train_df.head(10))

np.random.seed(1749)
sample_files = np.random.choice(os.listdir(base_path + train_dir), 50)
sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]
pd.options.display.width = 0
print(sample_df)

train_sample_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()


# train = shuffle(train)
# print(train.head(10))

# train_sample = train.reset_index(drop=True)  # df.reset_index(drop=True) drops the current index of the DataFrame and replaces it with an index of increasing integers. It never drops columns.
# print(train_sample.head(10))

# # Setting up the input images (x values or independent values)

# x_head = pd.DataFrame(train_sample, columns=["filename"])
# print(x_head.head(10))
# print(type(x_head))

# # Setting up the correponding values for images (y values or dependent values)

# y_values = pd.DataFrame(train_sample, columns=["any", "epidural", "intraventricular", "subarachnoid", "subdural"])
# print(y_values.head(10))
# print(type(y_values))

# ########### Rescale, Resize and Convert to PNG ###########


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
    save_dir = ("E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/")
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


save_and_resize(filenames=sample_files, load_dir='E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images/')


batch_size = 64


def create_datagen():
    return ImageDataGenerator(validation_split=0.15)


def create_test_gen():
    return ImageDataGenerator().flow_from_dataframe(
        submission_df,
        directory='E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_test_images/',
        x_col='filename',
        class_mode=None,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    )


def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        train_sample_df,
        directory='E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images/',
        x_col='filename',
        y_col=['any', 'epidural', 'intraparenchymal',
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='multi_output',
        target_size=(224, 224),
        batch_size=batch_size,
        subset=subset
    )


# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, 'training')

# val_gen = create_flow(data_generator, 'validation')

# test_gen = create_test_gen()

# # 4.0 ########### CREATE A DATA GENERATOR ###########


# # gen = ImageDataGenerator(
# #     rotation_range=20,
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     shear_range=0.1,
# #     zoom_range=0.2,
# #     horizontal_flip=True,
# #     vertical_flip=True,
# # )

# # train_generator = gen.flow_from_dataframe(
# #     train_path,
# #     target_size=img_size,
# #     shuffle=True,
# #     batch_size=batch_size,
# # )


# # # 5.0 ########### CREATE AND INITIALISE MODEL ###########


# # conv_base = ResNet50(weights='../input/models/model_weights_resnet.h5',
# #                      include_top=False,
# #                      input_shape=(q_size, q_size, img_channel))

# # conv_base.trainable = True

# # model = Sequential()
# # model.add(conv_base)
# # model.add(Flatten())
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(6, activation='sigmoid'))

# # # 6.0 ########### COMPILE MODEL ###########

# # model.compile(optimizer=RMSprop(lr=1e-5),
# #               loss='binary_crossentropy',
# #               metrics=['binary_accuracy'])

# # model.summary()

# # # 7.0 ########### TRAIN MODEL ON DATA PASSED THROUGH THE DATA GENERATOR ###########

# # history = model.fit_generator(generator=train_generator,
# #                               validation_data=val_generator,
# #                               epochs=epochs,
# #                               class_weight=class_weight,
# #                               workers=4)

# # # 8.0 ########### EVALUATION: PLOT LOSS VALUES ###########

# # loss = history.history['loss']
# # loss_val = history.history['val_loss']
# # epochs = range(1, len(loss) + 1)
# # plt.plot(epochs, loss, 'bo', label='loss_train')
# # plt.plot(epochs, loss_val, 'b', label='loss_val')
# # plt.title('value of the loss function')
# # plt.xlabel('epochs')
# # plt.ylabel('value of the loss functio')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # ########### EVALUATION: PLOT ACCURACY VALUES ###########

# # acc = history.history['binary_accuracy']
# # acc_val = history.history['val_binary_accuracy']
# # epochs = range(1, len(loss) + 1)
# # plt.plot(epochs, acc, 'bo', label='accuracy_train')
# # plt.plot(epochs, acc_val, 'b', label='accuracy_val')
# # plt.title('accuracy')
# # plt.xlabel('epochs')
# # plt.ylabel('value of accuracy')
# # plt.legend()
# # plt.grid()
# # plt.show()
