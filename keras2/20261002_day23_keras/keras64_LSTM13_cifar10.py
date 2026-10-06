# 39-3 카피
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#################### 데이터 증폭 ####################
datagen = ImageDataGenerator(
    rescale=1./255,   # . 표시 : 형변환 표시 안해도 되는데 해주는 편이 형변환했음을 보기 쉬움
    
    #### 데이터 증폭
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    # vertical_flip=True, # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    height_shift_range=0.1,
    rotation_range=15,    # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.1,
    shear_range=0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동. 찌부되는 거.
    fill_mode='nearest'
)

augment_size = 40000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 60000개 중 40000개 랜덤 뽑기
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)   # 60000개 중 40000개 랜덤 뽑기

x_augmented = x_train[randidx].copy()   # 메모리 공간 문제 방지를 위해 .copy() 사용함
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3
)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0] # [0]으로 x만 

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# scaling
x_train = x_train/255
x_test = x_test/255

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], -1, 3)
x_test = x_test.reshape(x_test.shape[0], -1, 3)
print(x_train.shape, x_test.shape)  # (90000, 1024, 3) (10000, 1024, 3)

#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(32,32,3))) # 30, 30, 64
# model.add(Conv2D(64, (3,3), activation='relu')) # 28,28,64
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # 14,14,64
# model.add(Conv2D(64, (3,3), activation='relu')) # 12,12,64
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu')) # 10,10,32
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # 5,5,32
# model.add(Conv2D(16, (3,3), activation='relu')) # 3,3,16
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(LSTM(64, input_shape=(1024, 3)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_13_", date, "-", filename])
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath = filepath
)
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
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
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 200/2000
# 562/563 [============================>.] - ETA: 0s - loss: 0.4695 - acc: 0.8422 
# Epoch 200: val_loss did not improve from 0.62170
# 563/563 [==============================] - 10s 19ms/step - loss: 0.4693 - acc: 0.8422 - val_loss: 0.6597 - val_acc: 0.7846
# 313/313 [==============================] - 2s 5ms/step - loss: 0.5814 - acc: 0.8088
# loss :  0.5814054608345032
# acc :  0.8087999820709229
# 313/313 [==============================] - 1s 4ms/step
# acc :  0.8088
# 걸린 시간 :  1838.87 초

# learning_rate = 0.01
# Epoch 86/2000
# 11/11 [==============================] - 0s 4ms/step - loss: 0.0105 - acc: 0.9941 - val_loss: 0.1655 - val_acc: 0.9912
# 4/4 [==============================] - 0s 1ms/step - loss: 0.8299 - acc: 0.9649
# loss :  0.8299
# acc :  0.9649
# 4/4 [==============================] - 0s 664us/step
# 걸린 시간 :  5.38 초
# acc_score : 0.9649122807017544

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
# Epoch 44: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
# 563/563 ━━━━━━━━━━━━━━━━━━━━ 7s 13ms/step - acc: 0.1004 - loss: 2.3029 - val_acc: 0.0998 - val_loss: 2.3033 - learning_rate: 0.0050
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - acc: 0.1000 - loss: 2.3027
# loss :  2.3027329444885254
# acc :  0.10000000149011612
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
# acc :  0.1
# 걸린 시간 :  336.76 초