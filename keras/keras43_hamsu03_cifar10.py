import time

import numpy as np
import pandas as pd

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping



#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# import matplotlib.pyplot as plt

# plt.imshow(x_train[4342],"gray")
# plt.show()

# # 스케일링 1(Minmax)
# x_train = x_train/255.
# x_test = x_test/255.

# 스케일링 2(MaxAbs)
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5
# DNN할 때
# x_train = x_train.reshape(-1, 32*32*3)
# x_test = x_test.reshape(-1, 32*32*3)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
model = Sequential()
# model.add(Dense(1024, activation='relu', input_shape= (32*32*3,)))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Conv2D(32, (3,3), input_shape=x_test[0].shape, activation="relu", padding="same"))
# model.add(Conv2D(64, (2,2), activation="relu"))
# model.add(Conv2D(32, (2,2), activation="relu"))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.2))

# # model.add(Flatten())
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=256, activation="relu"))
# model.add(Dense(128, activation="relu"))
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=64, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(units=16, activation="relu"))
# model.add(Dense(10, activation="softmax"))

input1 = Input(shape=x_train[0].shape)
conv2d1 = Conv2D(32, (3,3), activation="relu", padding="same")(input1)
conv2d2 = Conv2D(64, (2,2), activation="relu")(conv2d1)
conv2d3 = Conv2D(32, (2,2), activation="relu")(conv2d2)
drop1 = Dropout(0.2)(conv2d3)
conv2d4 = Conv2D(64, (2,2), activation="relu")(drop1)
maxpooling2d1 = MaxPooling2D((2,2))(conv2d4)
drop2 = Dropout(0.2)(maxpooling2d1)
conv2d5 = Conv2D(128, (2,2), activation="relu")(drop2)
maxpooling2d2 = MaxPooling2D((2,2))(conv2d5)
drop3 = Dropout(0.2)(maxpooling2d2)
gap2d1 = GlobalAveragePooling2D()(drop3)
dense1 = Dense(units=256, activation="relu")(gap2d1)
dense2 = Dense(128, activation="relu")(dense1)
dense3 = Dense(128, activation="relu")(dense2)
drop4 = Dropout(0.2)(dense3)
dense4 = Dense(units=64, activation="relu")(drop4)
dense5 = Dense(32, activation="relu")(dense4)
dense6 = Dense(units=16, activation="relu")(dense5)
output1 = Dense(10, activation="softmax")(dense6)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

batch_size = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=1, epochs=300, batch_size=batch_size, validation_split=0.2,
                    callbacks = [es]
                    )

train_time = time.time() - start_time


#4. 평가 예측
print("훈련 시간 :", round(train_time, 3), '초')
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print('acc :',acc)



#acc 0.92
# acc : 0.7316
# acc : 0.7239

# 훈련 시간: 134.751 초
# acc : 0.7358

# 훈련 시간(Flatten-e30): 171.24 초
# acc(Flatten-e30) : 0.7228
# 훈련 시간(GlobalAveragePooling2D): 134.497 초
# acc(GlobalAveragePooling2D) : 0.6615

# Epoch 28/300
# 훈련 시간(DNN-GPU): 34.343 초
# acc(DNN-GPU) : 0.5219

# Epoch 26/300
# 훈련 시간(DNN-CPU) : 104.89 초
# acc(DNN-CPU) : 0.5126

# Epoch 110/300
# 훈련 시간(함수형) : 667.099 초
# acc(함수형) : 0.7437