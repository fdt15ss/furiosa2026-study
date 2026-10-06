#23-2 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
import numpy as np
import datetime
import time

# acc = 0.95

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)
print(x, y)

y = pd.get_dummies(y)
print(y)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=52,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(13,1)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_8_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
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
    patience=100,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=8,
    verbose=1,
    callbacks=[es, mcp, rlr],
    validation_split=0.25
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1], 2))

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", round(acc, 4))

# Results

# RobustScaler
# Epoch 134/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 1.0000 - loss: 9.7202e-05 - val_acc: 0.9722 - val_loss: 0.0422
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - acc: 1.0000 - loss: 0.0331
# loss :  0.0331292599439621
# acc :  1.0
# 걸린 시간 :  9.78 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
# acc :  1.0

# learning_rate = 0.01
# Epoch 115/3000
# 14/14 [==============================] - 0s 3ms/step - loss: 5.3982e-08 - acc: 1.0000 - val_loss: 0.6030 - val_acc: 0.9722
# 2/2 [==============================] - 0s 1ms/step - loss: 0.0925 - acc: 0.9722
# loss :  0.09251780062913895
# acc :  0.97
# 걸린 시간 :  6.05 초
# 2/2 [==============================] - 0s 0s/step
# acc :  0.9722

# learning_rate = 0.0001
# Epoch 339/3000
# 14/14 [==============================] - 0s 3ms/step - loss: 0.0715 - acc: 0.9811 - val_loss: 0.0608 - val_acc: 0.9722
# 2/2 [==============================] - 0s 0s/step - loss: 0.1081 - acc: 0.9722
# loss :  0.10805962234735489
# acc :  0.97
# 걸린 시간 :  19.2 초
# 2/2 [==============================] - 0s 0s/step
# acc :  0.9722

# ReduceLROnPlateau
# Epoch 138: val_loss did not improve from 3637.18457
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 8491.7764 - val_loss: 4566.0269 - learning_rate: 6.2500e-04
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 4223.6133
# loss :  4223.61328125

# LSTM
# Epoch 139: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - acc: 1.0000 - loss: 0.0014 - val_acc: 0.8889 - val_loss: 0.9328 - learning_rate: 6.2500e-04
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - acc: 0.9167 - loss: 0.2511
# loss :  0.2510693371295929
# acc :  0.92
# 걸린 시간 :  40.05 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 121ms/step
# acc :  0.9167