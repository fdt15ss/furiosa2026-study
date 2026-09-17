import time

import numpy as np
import pandas as pd

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping



#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))   #([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                                #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                                #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                                #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                                #    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                                                #    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# import matplotlib.pyplot as plt
# plt.imshow(x_train[4342])
# plt.show()

# # # 스케일링 1(Minmax)
# x_train = x_train/255.
# x_test = x_test/255.

# 스케일링 2(MaxAbs)
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5
x_train = x_train.reshape(-1, 32*32*3)
x_test = x_test.reshape(-1, 32*32*3)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(32*32*3,)))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), input_shape=x_test[0].shape, activation="relu"))
# model.add(Conv2D(128, (2,2), activation="relu"))
# model.add(Conv2D(128, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.3))
# model.add(Conv2D(256, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.3))

# # model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="softmax"))

model.summary()
# exit()
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
print("훈련 시간 :", round(train_time,3), '초')
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print('acc :', acc)


#acc 0.92
# acc : 0.2579
# acc : 0.4094

# 훈련 시간(Flatten-ep30) : 232.454 초
# acc(Flatten-ep30) : 0.4284

# 훈련 시간(GlobalAveragePooling2D-ep30) : 238.316 초
# acc(GlobalAveragePooling2D-ep30) : 0.375

# Epoch 33/300
# 훈련 시간(DNN-GPU) : 46.861 초
# acc(DNN-GPU) : 0.192

# Epoch 33/300
# 훈련 시간(DNN-CPU) : 353.801 초
# acc(DNN-CPU) : 0.1897