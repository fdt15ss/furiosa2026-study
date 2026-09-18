import time

import numpy as np
import pandas as pd

from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping



#1. 데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# import matplotlib.pyplot as plt

# plt.imshow(x_train[4342],"gray")
# plt.show()

# 스케일링 2(MaxAbs)
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
# DNN 할 때
# x_train = x_train.reshape(-1,28*28*1)
# x_test = x_test.reshape(-1,28*28*1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
# model = Sequential()
# model.add(Dense(1024, activation="relu", input_shape=(28*28*1,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))

# model.add(Conv2D(32, (3,3), input_shape=x_test[0].shape, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation="relu"))
# model.add(Dropout(0.3))
# model.add(Conv2D(128, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(256, (2,2), activation="relu"))
# model.add(Dropout(0.2))

# # model.add(Flatten())
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=256, activation="relu"))
# model.add(Dropout(0.3))
# model.add(Dense(units=128, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(16, activation="relu"))
# model.add(Dense(10, activation="softmax"))

#2-2. 함수형 모델
input1 = Input(shape=x_train[0].shape)
conv2d1=(Conv2D(32, (3,3), activation="relu"))(input1)
drop1=(Dropout(0.2))(conv2d1)
conv2d2=(Conv2D(64, (3,3), activation="relu"))(drop1)
drop2=(Dropout(0.3))(conv2d2)
conv2d3=(Conv2D(128, (2,2), activation="relu"))(drop2)
maxpooling2d1=(MaxPooling2D((2,2)))(conv2d3)
drop3=(Dropout(0.2))(maxpooling2d1)
conv2d4=(Conv2D(128, (2,2), activation="relu"))(drop3)
maxpooling2d2=(MaxPooling2D((2,2)))(conv2d4)
conv2d5=(Conv2D(256, (2,2), activation="relu"))(maxpooling2d2)
drop4=(Dropout(0.2))(conv2d5)

gap2d1=(GlobalAveragePooling2D())(drop4)
dense1=(Dense(units=256, activation="relu"))(gap2d1)
drop1=(Dropout(0.3))(dense1)
dense2=(Dense(units=128, activation="relu"))(drop1)
drop2=(Dropout(0.2))(dense2)
dense3=(Dense(64, activation="relu"))(drop2)
dense4=(Dense(32, activation="relu"))(dense3)
dense5=(Dense(16, activation="relu"))(dense4)
output1=(Dense(10, activation="softmax"))(dense5)
model = Model(inputs= input1, outputs= output1)

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
print('훈련 시간 :', round(train_time,3), '초')
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print('acc :', acc)



#acc 0.92
#0.9181 

# 훈련 시간 : 252.688 초
# acc : 0.9157

# 훈련 시간(Flatten-e30) : 129.38 초
# acc(Flatten-e30) : 0.9191

# 훈련 시간(GlobalAveragePooling2D-e30) : 134.894 초
# acc(GlobalAveragePooling2D-e30) : 0.912

# 훈련 시간(DNN-GPU) : 35.863 초
# acc(DNN-GPU) : 0.8871

# 훈련 시간(DNN-CPU) : 128.345 초
# acc(DNN-CPU) : 0.8843

# Epoch 65/300
# 훈련 시간(함수형) : 282.535 초
# acc(함수형) : 0.9206