import time

import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping


#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train[3])
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
# print(pd.value_counts(y_test))
# print(np.max(x_train),np.min(x_train)) #255 0
# print(np.max(x_test),np.min(x_test)) #255 0

# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.
# print(np.max(x_train),np.min(x_train))    #1.0 0.0
# print(np.max(x_test),np.min(x_test))      #1.0 0.0

# 스케일링 2(MaxAbs)
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train),np.min(x_train))  #1.0 -1.0
# print(np.max(x_test),np.min(x_test))    #1.0 -1.0

x_train = x_train.reshape(-1,28*28*1)
x_test = x_test.reshape(-1,28*28*1)
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

# exit()


ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_shape=(28*28,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Conv2D(64, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu")) #(24, 24, 32)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation="relu"))
# model.add(Conv2D(16, (2,2), activation="relu"))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation="relu"))
# model.add(MaxPooling2D((2,2)))

# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation="relu")) #(20,20,16)

# # model.add(Flatten()) #(None, 6400)
# model.add(GlobalAveragePooling2D())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

batch_size = 64
start_time = time.time()

history = model.fit(x_train,y_train,
                    verbose=1,
                    epochs=300,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks = [es],
                    )

train_time = time.time() - start_time


#4. 평가 예측
print("총 예측 :", round(train_time, 5), "초")
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(acc)

# 0.9917

# 총 예측 : 152.23938 초
# 0.9922

# 총 예측 : 205.9731 초
# 0.9868

# 총 예측(Flatten-30) : 106.08722 초
# 0.9904

# 총 예측(GlobalAveragePooling2D-30) : 99.50662 초
# 0.9845

# 총 예측(DNN-GPU) : 120.92317 초
# 0.9759

# 총 예측(DNN-CPU) : 374.53665 초
# 0.9819