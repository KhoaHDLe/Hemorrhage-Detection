
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import pydicom
import matplotlib.pylab as plt
from matplotlib import rcParams
import os

rcParams['figure.figsize'] = 11.7, 8.27

train_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images"
test_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data-Data/stage_1_test_images"

train_csv = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_train.csv")

print(train_csv.head(10))

train_csv["type"] = train_csv["ID"].apply(lambda x: x.split("_")[2])
train_csv["ID"] = train_csv["ID"].apply(lambda x: ".".join(x.split("_")[0:2]))

print(train_csv.head(10))

sns.countplot(train_csv.Label)
plt.show()

lable_group = train_csv.groupby("type").sum()
print(lable_group)
