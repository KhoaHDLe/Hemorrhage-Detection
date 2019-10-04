
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import pydicom  # to read medical images
import matplotlib.pylab as plt
from matplotlib import rcParams
import os
import operator

rcParams['figure.figsize'] = 11.7, 8.27

train_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images"
test_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data-Data/stage_1_test_images"

# Reading in CSV files

train_csv = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_train.csv")
sample_csv = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_sample_submission.csv")

print(train_csv.head(10))
print(sample_csv.head(10))

print("Train shape : {}".format(train_csv.shape))
print("Test shape : {}".format(sample_csv.shape))

# Rearranging CSV tables for manipulation

train_csv["Type"] = train_csv["ID"].apply(lambda x: x.split("_")[2])
train_csv["ID"] = train_csv["ID"].apply(lambda x: "_".join(x.split("_")[0:2]))
train_csv["Filename"] = train_csv["ID"].apply(lambda x: "_".join(x.split("_")[0:2]) + ".dcm")

sample_csv["Type"] = sample_csv["ID"].apply(lambda x: x.split("_")[2])
sample_csv["ID"] = sample_csv["ID"].apply(lambda x: "_".join(x.split("_")[0:2]))
sample_csv["Filename"] = sample_csv["ID"].apply(lambda x: "_".join(x.split("_")[0:2]) + ".dcm")

print(train_csv.head(13))
print(sample_csv.head(13))

print('There are {} images in the train data set'.format(len(train_csv.ID.unique())))
print('There are {} images in the test data set'.format(len(sample_csv.ID.unique())))

# Number of unique labels (Hemorrhage =1  or no Hemorrhage =0)

value_dict = train_csv.Label.value_counts().to_dict()
print(value_dict)
print(list(value_dict.keys()))
print(list(value_dict.values()))

fig, ax = plt.subplots()
sns.barplot(x=list(value_dict.keys()), y=list(value_dict.values()))
ax.set_title("the number of labels")
ax.set_xlabel("class")
plt.show()

print('{:.1f} % of images have at least one type of Hemorrhage.'.format((value_dict[1] / value_dict[0]) * 100))

# Type and number of Hemorrhages. Remember an image may have more than one type of Hemorrhage.

type_dict = train_csv[train_csv['Label'] == 1].Type.value_counts().to_dict()

fig, ax = plt.subplots(figsize=(18, 6))

sns.countplot(x="Type", hue="Label", data=train_csv, palette="ch:.25")

ax.set_title("Count of Hemorrhages by Type")
ax.legend(title='Classification', bbox_to_anchor=(1, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
ax.set_xlabel("type")
ax.set_ylabel("Count")

for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x() + 0.12, p.get_height() + 5000), fontsize=8)

plt.show()

# fig, ax = plt.subplots(figsize = (18,6))

# sns.countplot(x=list(type_dict.keys()), y=list(type_dict.values()))
# ax.set_title("the number of different type of Hemorrhage")
# ax.set_xlabel("type")
# plt.show()


for index, count in type_dict.items():
    print('There are {} instances of {} hemorrage types in the train data set'.format(count, index))

# The number of Hemorrhages in an image distribution

images_count = train_csv[(train_csv['Label'] == 1) & (train_csv['Type'] != 'any')].pivot_table(values="Label", index=["ID"], aggfunc='sum').Label.value_counts().to_dict()

fig, ax = plt.subplots()
sns.barplot(x=list(images_count.keys()), y=list(images_count.values()))
ax.set_title("the number of images with different different class of Hemorrhage")
ax.set_xlabel("number of Hemorrhage")
plt.show()

for index, count in images_count.items():
    print('There are {} images that have {} hemorrage type in the train data set'.format(count, index))
