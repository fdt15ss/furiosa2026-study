# 50-2 카피
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import datetime
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


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
# print(x_train.shape[0]) # 60000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 60000개 중 40000개 랜덤 뽑기
# print(randidx.shape)    # ndarray라 shape로 갯수 셈
# print(len(randidx)) # list, tuple은 len으로 갯수 셈

randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)   # 60000개 중 40000개 랜덤 뽑기
# randint 쓰면 중복있을 수 있어서 choice로 바꿈. replace(defalut는 True) Fasle로 바꿔줘야 랜덤이라고 해서 넣음.

# print(np.min(randidx), np.max(randidx)) # 1 59992. 0~59999 범위

x_augmented = x_train[randidx].copy()   # 메모리 공간 문제 방지를 위해 .copy() 사용함
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)
# exit()

# x_augmented = x_augmented.reshape(-1, 28, 28, 1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 1
)
# print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0] # [0]으로 x만 

# print(x_augmented.shape)    # (40000, 28, 28, 1)

# print(x_train.shape)    # (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# scale
x_train = x_train / 255
x_test = x_test / 255

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([10057, 10184,  9940, 10005,  9910, 10012,  9951, 10019, 10005, 9917], dtype=int64))


# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# # one-hot encoding
# ohe = OneHotEncoder(sparse_output=False)

# y_train = ohe.fit_transform(y_train.reshape(-1,1))
# y_test = ohe.fit_transform(y_test.reshape(-1,1))

# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # 24,24,64
model.add(Conv2D(64, (5,5), activation='relu')) # 20, 20, 64
model.add(Dropout(0.2))
model.add(MaxPooling2D())   # 10, 10, 64
model.add(Conv2D(32, (3,3), activation='relu')) # 8,8,32
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu')) # 6,6,32
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

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
filepath = "".join([path, "k53_12_", date, "-", filename])

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
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
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
# y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 139: val_loss did not improve from 0.38492
# 625/625 [==============================] - 4s 7ms/step - loss: 0.1602 - acc: 0.9390 - val_loss: 0.3961 - val_acc: 0.8605
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2381 - acc: 0.9192
# loss :  0.23805087804794312
# acc :  0.9192000031471252
# 313/313 [==============================] - 0s 1ms/step
# acc :  0.9192
# 걸린 시간 :  566.68 초

# ReduceLROnPlateau
# Epoch 56: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 6s 9ms/step - acc: 0.0980 - loss: 2.3030 - val_acc: 0.0992 - val_loss: 2.3032 - learning_rate: 0.0050
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - acc: 0.7678 - loss: 0.5817
# loss :  0.5817225575447083
# acc :  0.767799973487854
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
# acc :  0.7678
# 걸린 시간 :  340.57 초