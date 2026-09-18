import time
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import AveragePooling2D, Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D

from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint

import my_util

path = "./_save/catdog/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k44_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터

train_datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    rescale=1./255,
    # horizontal_flip=True,   # 수평 반전
    # vertical_flip=True,     # 수직 반전
    # width_shift_range= 0.1, # 평형 이동
    # height_shift_range=0.1, # 수직 이동
    # rotation_range= 5,      # 각도 조절
    # zoom_range=1.2,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    # fill_mode="nearest",
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train="./_data/image/catdog/training_set/"
path_test="./_data/image/catdog/test_set/"

xy_train = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(100,100),
    color_mode="rgb",
    batch_size=2300,
    class_mode="binary",
    shuffle=True

)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    color_mode="rgb",
    batch_size=2500,
    class_mode="binary",
    # shuffle=False
)

x_train = np.concatenate((xy_train[0][0], xy_train[1][0],xy_train[2][0],xy_train[3][0]),axis=0)
y_train = np.concatenate((xy_train[0][1], xy_train[1][1],xy_train[2][1],xy_train[3][1]),axis=0)

x_test = xy_test[0][0]
y_test = xy_test[0][1]



# print(x_train.shape ,y_train.shape)
# print(x_test.shape ,y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu")) #(24, 24, 32)
# model.add(Dropout(0.1))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
# model.add(Dropout(0.1))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(AveragePooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(64, (2,2), activation="relu")) #(20,20,16)
model.add(Conv2D(16, (2,2), activation="relu")) #(20,20,16)
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(32, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(AveragePooling2D())
model.add(Conv2D(128, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D()) #(None, 6400)
model.add(Dense(units=64, activation="relu"))
# model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(1,activation="sigmoid"))
# model.summary()
# exit()

#3. 컴파일 훈련
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

batch_size = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=5000, batch_size=batch_size, validation_split=0.25, callbacks = [es,mcp])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) #accuracy_score 에서 필요
# print(y_pred[:10])

acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss[0],
    sub_score = acc,
    train_ration = 0,
    csv_file_path="catdog.csv"
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")