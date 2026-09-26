import time

import numpy as np
import pandas as pd

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import my_util


#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))   #([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                                #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                                #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                                #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                                #    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,

# 데이터 증폭
datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    rescale=1./255,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,   # 상하 반전
    width_shift_range=0.1,   # 평형 이동
    height_shift_range=0.1,  # 수직 이동
    rotation_range=20,      # 각도 조절
    zoom_range=0.1,         #
    # shear_range=0.7,      # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest"
)

augment_size = 40000

# xy_data = datagen.flow(
#     np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),
#     np.zeros(augment_size),
#     batch_size=augment_size,
#     shuffle=False,
# ).next()

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = datagen.flow(
    x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3),
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# 데이터 변환 완료. 데이터 합병
x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(-1,x_train.shape[1],x_train.shape[2],3)

x_train = np.concatenate((x_train/255.,x_augmented))
y_train = np.concatenate((y_train,y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
x_test = x_test/255.

print(np.unique(y_train, return_counts=True))
                                                #    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# exit()
# import matplotlib.pyplot as plt
# plt.imshow(x_train[4342])
# plt.show()

# # # 스케일링 1(Minmax)
# x_train = x_train/255.
# x_test = x_test/255.

# 스케일링 2(MaxAbs)
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=x_train[0].shape, activation="relu"))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(Dropout(0.3))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(100, activation="softmax"))

#3. 컴파일 훈련
from keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001     # 디폴트
learning_rate = 0.002
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = "categorical_crossentropy",
              optimizer = Adam(learning_rate=learning_rate),
              metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode = 'auto',
    patience = 5,
    verbose=1,
    factor=0.5
)

BATCH_SIZE = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=1, epochs=2000,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    callbacks = [es, rlr]
                    )

train_time = time.time() - start_time


#4. 평가 예측
print('걸린시간 : ', round(train_time,3), '초')
loss, acc_eval = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = BATCH_SIZE,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = acc,
    train_ration = 0,
    csv_file_path="14_cifar100.csv"
)

