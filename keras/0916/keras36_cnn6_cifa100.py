import time

import numpy as np
import pandas as pd

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

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

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=x_test[0].shape, activation="relu"))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(Conv2D(256, (2,2), activation="relu"))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(100, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 50, restore_best_weights= True, 
)

batch_size = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=2000, batch_size=batch_size, validation_split=0.2, callbacks = [es])

train_time = time.time() - start_time


#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score = acc,
    train_ration = 0,
    csv_file_path="cifar100.csv"
)





#acc 0.92