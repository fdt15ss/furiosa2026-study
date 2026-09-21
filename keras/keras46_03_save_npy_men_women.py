# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from datetime import datetime
import my_util
from sklearn.model_selection import train_test_split

traintest_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,   # 수평 뒤집기,
    # vertical_flip=True,     # 수직 뒤집기 (상하 반전)
    # width_shift_range=0.1,  # 평행이동
    # height_shift_range=0.1,
    # rotation_range=5,   # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7,    # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # fill_mode = 'nearest'
)

# test_datagen = ImageDataGenerator( # 시험지는 그대로 둬야함. 테스트 데이터는 변환하지 않음. 
#     rescale = 1./255,
# )

path_set = "./_data/image/men_women/"

xy = traintest_datagen.flow_from_directory(
   path_set, # 경로
   target_size=(200, 200),
   batch_size=10000, # 30000?
   class_mode='binary', # 이진분류
   color_mode='rgb', 
   shuffle=True,
)
# 

print(xy[0][0].shape) # (160, 150, 150, 1)
print(xy[0][1].shape) # (160,)

x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42, stratify=y)

print(x_train.shape, y_train.shape) # (8000, 200, 200, 3) (8000,)
print(x_test.shape, y_test.shape)   # (2000, 200, 200, 3) (2000,)

npy_path = '_data\\men_women_npy\\'
np.save(npy_path + 'keras46_men_women_x_train.npy', x_train)
np.save(npy_path + 'keras46_men_women_y_train.npy', y_train)
np.save(npy_path + 'keras46_men_women_x_test.npy', x_test)
np.save(npy_path + 'keras46_men_women_y_test.npy', y_test)