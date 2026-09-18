# 44-2 카피

# https://drive.google.com/drive/folders/1XnXBR2rdrHf9GLEgMyDFq9Ellp5DeX2P

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import time

train_datagen = ImageDataGenerator(
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
test_datagen = ImageDataGenerator( # 시험지는 그대로 둬야함. 테스트 데이터는 변환하지 않음. 
    rescale = 1./255,
)

path_train = "./_data/image/brain/train/"
path_test = "./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
   path_train, # 경로
   target_size=(150, 150),
   batch_size=160,
   class_mode='binary', # 이진분류
   color_mode='grayscale', #흑백
   shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size = (150, 150),
    batch_size=120,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False, # test에서는 필요가 없다.
)
# Found 120 images belonging to 2 classes.
print(xy_train[0][0].shape) # (160, 150, 150, 1)
print(xy_train[0][1].shape) # (160,)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)

np_path = "./_data/kaggle_cat_dog_npy/"
np.save(np_path + 'keras45_01_x_train.npy', arr= xy_train[0][0]) # x_train
np.save(np_path + 'keras45_01_y_train.npy', arr= xy_train[0][1]) # y_train
np.save(np_path + 'keras45_01_x_test.npy', arr= xy_test[0][0]) # x_test
np.save(np_path + 'keras45_01_y_test.npy', arr= xy_test[0][1]) # y_test
