
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import pydicom  # to read medical images
import matplotlib.pylab as plt
from matplotlib import rcParams
import os
import operator

# Specifiy directories
train_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data/stage_1_train_images"
test_dir = "E:/4.0 Projects/Hemorrhage-Detection/Hemorrhage-Detection-Data-Data/stage_1_test_images"

# Reading in CSV files
train_csv = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_train.csv")
sample_csv = pd.read_csv("E:/4.0 Projects/Hemorrhage-Detection/stage_1_sample_submission.csv")

# Rearranging CSV tables for manipulation

train_csv["Type"] = train_csv["ID"].apply(lambda x: x.split("_")[2])
train_csv["ID"] = train_csv["ID"].apply(lambda x: "_".join(x.split("_")[0:2]))
train_csv["Filename"] = train_csv["ID"].apply(lambda x: "_".join(x.split("_")[0:2]) + ".dcm")

filename = pydicom.read_file(os.path.join(train_dir, "ID_000039fa0.dcm"))
print(filename)

# pivot = train_csv[train_csv['Type'] != 'any'].pivot_table(values='Label', index=['ID'], aggfunc='sum').reset_index()

# print(train_csv.ID.tolist())


# fig = plt.figure()
# for j in range(4):
#     for i, image in enumerate(pivot[pivot.Label == 1].iloc[j * 4:(j + 1) * 4, ].Filename.tolist()):
#         ax = fig.add_subplot(4, 4, j * 4 + i + 1, xticks=[], yticks=[])
#         img = np.array(pydicom.read_file(os.path.join(train_dir, image)).pixel_array)
#         img = exposure.equalize_hist(img)
#         plt.imshow(img, cmap=plt.cm.bone)
#         ax.set_title('One Hemorrhage ' + image)
