from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import datetime
import time
#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000,
    # maxlen=100,
)

# print(x_train)
# [list([1, 14, 22, 16, 43, 530, 973, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 2, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2, 2, 16, 480, 66, 2, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 2, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32])
#  ...
#   list([1, 17, 6, 194, 337, 7, 4, 204, 22, 45, 254, 8, 106, 14, 123, 4, 2, 270, 2, 5, 2, 2, 732, 2, 101, 405, 39, 14, 2, 4, 2, 9, 115, 50, 305, 12, 47, 4, 168, 5, 235, 7, 38, 111, 699, 102, 7, 4, 2, 2, 9, 24, 6, 78, 2, 17, 2, 2, 21, 27, 2, 2, 5, 2, 2, 92, 2, 4, 2, 7, 4, 204, 42, 97, 90, 35, 221, 109, 29, 127, 27, 118, 8, 97, 12, 157, 21, 2, 2, 9, 6, 66, 78, 2, 4, 631, 2, 5, 2, 272, 191, 2, 6, 2, 8, 2, 2, 2, 544, 5, 383, 2, 848, 2, 2, 497, 2, 8, 2, 2, 2, 21, 60, 27, 239, 9, 43, 2, 209, 405, 10, 10, 12, 764, 40, 4, 248, 20, 12, 16, 5, 174, 2, 72, 7, 51, 6, 2, 22, 4, 204, 131, 9])]

# print(y_train)  # [1 0 0 ... 0 1 0]
# print(np.unique(y_train, return_counts=True))   # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
# print(x_train.shape, y_train.shape) # (25000,) (25000,)
# print(x_test.shape, y_test.shape)   # (25000,) (25000,)

# print("x_train max len : ", max(len(i) for i in x_train))   # x_train max len :  2494
# print("x_train min len : ", min(len(i) for i in x_train))   # x_train min len :  11
# print("x_train average len : ", sum(map(len, x_train))/len(x_train))    # x_train average len :  238.71364

# pad_sequences
x_train = pad_sequences(
    x_train,
    padding='pre',
    maxlen=239,
    truncating='post'
)
x_test = pad_sequences(
    x_test,
    padding='pre',
    maxlen=239,
    truncating='post'
)

# acc 0.6

#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=1000))
# model.add(LSTM(100, input_shape=(146, 1000)))  # Embedding에서 input_length가 146이고 output이 (N, 146, 1000)이므로 (N, 146, 1000) -> (146, 1000).
# input_shape에서 146으로 안 적고 다르게 적어도 에러 안 나길래 찾아보니 알아서 잘 처리해준다고 함.
model.add(Bidirectional(LSTM(100), input_shape=(239,1000)))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

path = './_save/keras62/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path + "keras62_2_" + date + "-" + filename])

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.5
)

learning_rate = 0.0005

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=32,
    validation_split=0.25,
    verbose=1,
    callbacks = [es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_train, y_train)
print("loss : ", results[0])
print("acc : ", results[1])

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, np.round(y_pred))
print("x_test의 예측 acc : ", acc)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 3: val_loss improved from 0.35853 to 0.34063, saving model to ./_save/keras62\keras62_2_1002_1247-0003-0.3406.keras
# 586/586 [==============================] - 16s 28ms/step - loss: 0.3254 - acc: 0.8591 - val_loss: 0.3406 - val_acc: 0.8534 - lr: 0.0010
# 782/782 [==============================] - 9s 12ms/step - loss: 0.2821 - acc: 0.8846
# loss :  0.2820592522621155
# acc :  0.8845999836921692
# 782/782 [==============================] - 9s 11ms/step
# x_test의 예측 acc :  0.85
# 걸린 시간 :  52.55 초

# Epoch 54: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
# 586/586 [==============================] - 16s 28ms/step - loss: 0.0091 - acc: 0.9966 - val_loss: 1.6940 - val_acc: 0.8206 - lr: 0.0010
# Epoch 54: early stopping
# 782/782 [==============================] - 9s 11ms/step - loss: 0.2805 - acc: 0.8848
# loss :  0.2804974913597107
# acc :  0.8848000168800354
# 782/782 [==============================] - 9s 11ms/step
# x_test의 예측 acc :  0.84956
# 걸린 시간 :  891.16 초