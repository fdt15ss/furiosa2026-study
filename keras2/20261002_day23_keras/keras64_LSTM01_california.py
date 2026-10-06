# 52-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
# import matplotlib.pyplot as plt
import time

path = './_save/keras52_01/'


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(20, input_dim=8))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1))

model = Sequential()
model.add(LSTM(20, input_shape=(8, 1)))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 50,    # 임계값
    restore_best_weights = True,  # 디폴트는 False. False면 스탑 됐을 때의 가중치를 반환.,
    verbose=1
)

path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_1_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=3000, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, mcp, rlr])

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

# Results

# learning_rate = 0.01
# Epoch 73/3000
# 387/387 [==============================] - ETA: 0s - loss: 0.4746Restoring model weights from the end of the best epoch: 23.
# 387/387 [==============================] - 1s 2ms/step - loss: 0.4746 - val_loss: 0.4896
# Epoch 73: early stopping
# 129/129 [==============================] - 0s 725us/step - loss: 0.4084
# loss :  0.4084179699420929
# 걸린 시간 :  44.64 초

# learning_rate = 0.0001
# Epoch 277/3000
# 371/387 [===========================>..] - ETA: 0s - loss: 0.4262Restoring model weights from the end of the best epoch: 227.
# 387/387 [==============================] - 1s 2ms/step - loss: 0.4268 - val_loss: 0.4473
# Epoch 277: early stopping
# 129/129 [==============================] - 0s 760us/step - loss: 0.4086
# loss :  0.40856730937957764
# 걸린 시간 :  169.85 초

# ReduceLROnPlateau
# Epoch 254/3000
# 370/387 [===========================>..] - ETA: 0s - loss: 0.3992Restoring model weights from the end of the best epoch: 204.
# 387/387 [==============================] - 3s 8ms/step - loss: 0.3986 - val_loss: 0.4041 - lr: 1.5625e-04
# Epoch 254: early stopping
# 129/129 [==============================] - 0s 831us/step - loss: 0.3915
# loss :  0.3915115296840668
# 걸린 시간 :  529.72 초

# LSTM
# Epoch 65/3000
# 384/387 [============================>.] - ETA: 0s - loss: 0.2635Restoring model weights from the end of the best epoch: 15.
# 387/387 [==============================] - 1s 3ms/step - loss: 0.2630 - val_loss: 0.5341 - lr: 0.0025
# Epoch 65: early stopping
# 129/129 [==============================] - 0s 1ms/step - loss: 0.4336
# loss :  0.4336012005805969
# 걸린 시간 :  71.21 초