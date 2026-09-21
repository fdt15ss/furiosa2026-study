import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from datetime import datetime
import my_util
import pandas as pd
from sklearn.model_selection import train_test_split

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

path = "./_data/image/horse-human/"

xy = train_datagen.flow_from_directory(
   path, # 경로
   target_size=(200, 200),
   batch_size=10000,
   class_mode='categorical', # 이진분류
   color_mode='rgb', 
   shuffle=True,
)
# Found 160 images belonging to 2 classes.



# Found 120 images belonging to 2 classes.
print(xy[0][0].shape) # (1027, 200, 200, 3)
print(xy[0][1].shape) # (1027, 2)

x = xy[0][0]
y = xy[0][1]

print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# y_cat = to_categorical(y)(이미지데이터 제네레이터가 이미 카테고리로 y 값을 만들어 놓음)  
# y = pd.get_dummies(y, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)

# exit()
npy_path = './_data/horse-human_npy/'

np.save(npy_path + 'keras46_horse-human_x_train.npy', x_train)
np.save(npy_path + 'keras46_horse-human_y_train.npy', y_train)
np.save(npy_path + 'keras46_horse-human_x_test.npy', x_test)
np.save(npy_path + 'keras46_horse-human_y_test.npy', y_test)