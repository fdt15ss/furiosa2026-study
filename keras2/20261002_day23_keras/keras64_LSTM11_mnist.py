# 36-2 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test)) # 255 0

###### 스케일링 1
x_train = x_train/255.  # python 기초 - . 붙여서 실수형으로 변환 (. 붙이면 .0으로 판단하고 실수로 처리함)
x_test = x_test/255   # python3부터는 정수 나눗셈하면 기본적으로 실수 나온다고 함
# 이미지 데이터인 거 아니까 그냥 255로 나누기

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test)) # 1.0 0.0

###### 스케일링 2
# x_train = (x_train-127.5)/127.5 # -1~1로 스케일링. 나눈 다음 그 값에서 1빼는 식으로 해도 됨
# x_test = (x_test-127.5)/127.5

# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape)    # (60000, 28, 28, 1)    # 4차원 형식되게 reshape
print(x_test.shape) # (10000, 28, 28, 1)

# exit()

######## One-hot Encoding ######
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_train = encoder.fit_transform(y_train)

y_test = y_test.reshape(-1,1)
y_test = encoder.fit_transform(y_test)

print(y_train.shape)    # (60000,10)
print(y_test.shape) # (10000,10)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
print(x_train.shape, x_test.shape)  # (60000, 784, 1) (10000, 784, 1)

#2. 모델 구성
model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))   # (26, 26, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # (24,24,32). 파라미터 이름도 적어봄
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation='relu')) # (23,23,32)
# model.add(Conv2D(16, (2,2), activation='relu')) # (22,22,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu')) # (21,21,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu')) # (20,20,16)
# model.add(Flatten())    # (6400,)
# # 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
# model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
# model.add(Dropout(0.2))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(10, activation='softmax'))  # (10,)

# model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
# model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # (20,20,64). 파라미터 이름도 적어봄
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (5,5), activation='relu')) # (16,16,32)
# model.add(Conv2D(32, (5,5), activation='relu')) # (12,12,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (5,5), activation='relu')) # (8,8,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (5,5), activation='relu')) # (4,4,32)
# model.add(Dropout(0.2))
# model.add(Flatten())    # (512,)
# # 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
# model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
# model.add(Dropout(0.2))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(10, activation='softmax'))  # (10,)

model.add(LSTM(64, input_shape=(784,1)))
model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # (10,)

model.summary()

#3. 컴파일, 훈련
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)

path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_11_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath = filepath
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
print("=========model.evaluate==========")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
# print(y_pred[:20])
# print(y_pred.shape)

# print(y_test[:20])
# print(y_test.shape)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# print(y_pred[:20])
# print(y_pred.shape)

# print(y_test[:20])
# print(y_test.shape)

acc = accuracy_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results

# parameter tuning
# Epoch 56/2000
# 375/375 [==============================] - 3s 9ms/step - loss: 0.0214 - acc: 0.9951 - val_loss: 0.0598 - val_acc: 0.9914
# =========model.evaluate==========
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0274 - acc: 0.9940 
# loss :  0.027361255139112473
# acc :  0.9940000176429749
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.994
# 걸린 시간 :  192.05 초

# learning_rate = 0.01
# Epoch 53: val_loss did not improve from 2.30187
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 30s 80ms/step - acc: 0.1140 - loss: 2.3017 - val_acc: 0.1060 - val_loss: 2.3027
# =========model.evaluate==========
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - acc: 0.1135 - loss: 2.3017 
# loss :  2.3016746044158936
# acc :  0.11349999904632568
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step  
# accuracy_score :  0.1135
# 걸린 시간 :  1599.5 초

# learning_rate = 0.0001
# Epoch 477/2000
# 11/11 [==============================] - 0s 4ms/step - loss: 0.0227 - acc: 0.9912 - val_loss: 0.0110 - val_acc: 1.0000
# 4/4 [==============================] - 0s 1ms/step - loss: 0.1794 - acc: 0.9737
# loss :  0.1794
# acc :  0.9737
# 4/4 [==============================] - 0s 0s/step
# 걸린 시간 :  32.36 초
# acc_score : 0.9736842105263158

# ReduceLROnPlateau
# Epoch 64: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 5s 13ms/step - acc: 0.1140 - loss: 2.3012 - val_acc: 0.1060 - val_loss: 2.3022 - learning_rate: 0.0025
# =========model.evaluate==========
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - acc: 0.1135 - loss: 2.3012
# loss :  2.301241636276245
# acc :  0.11349999904632568
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step
# accuracy_score :  0.1135
# 걸린 시간 :  330.99 초

# LSTM
# Epoch 67: val_loss did not improve from 2.30183
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 12s 32ms/step - acc: 0.1140 - loss: 2.3012 - val_acc: 0.1060 - val_loss: 2.3020 - learning_rate: 0.0025
# =========model.evaluate==========
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - acc: 0.1135 - loss: 2.3014
# loss :  2.3014187812805176
# acc :  0.11349999904632568
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step
# accuracy_score :  0.1135
# 걸린 시간 :  803.73 초