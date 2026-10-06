# 23-3카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import datetime

# acc = 0.93

# 1. 내부 연산(Matrix Multiplication 등)에 사용할 스레드 수
tf.config.threading.set_intra_op_parallelism_threads(16)

# 2. 독립적인 연산들을 병렬로 처리할 스레드 수
tf.config.threading.set_inter_op_parallelism_threads(16)

#1. 데이터
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

print(x, y)
# [[2.596e+03 5.100e+01 3.000e+00 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.590e+03 5.600e+01 2.000e+00 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.804e+03 1.390e+02 9.000e+00 ... 0.000e+00 0.000e+00 0.000e+00]
#  ...
#  [2.386e+03 1.590e+02 1.700e+01 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.384e+03 1.700e+02 1.500e+01 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.383e+03 1.650e+02 1.300e+01 ... 0.000e+00 0.000e+00 0.000e+00]] [5 5 2 ... 3 3 3]
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y), np.unique(y).size)  # [1 2 3 4 5 6 7] 7
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y)
print(y.shape)  # (581012, 7)
# print(y[:10])
#        1      2      3      4      5      6      7
# 0  False  False  False  False   True  False  False
# 1  False  False  False  False   True  False  False
# 2  False   True  False  False  False  False  False
# 3  False   True  False  False  False  False  False
# 4  False  False  False  False   True  False  False
# 5  False   True  False  False  False  False  False
# 6  False  False  False  False   True  False  False
# 7  False  False  False  False   True  False  False
# 8  False  False  False  False   True  False  False
# 9  False  False  False  False   True  False  False

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
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
model.add(LSTM(50, input_shape=(54,1)))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_9_", date, "-", filename])

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
hist = model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=128,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1], 4))

print("걸린 시간 : ", round(end_time-start_time), 2, "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", round(acc, 4))

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='blue', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.plot(hist.history['acc'], c='orange', label='acc')
plt.plot(hist.history['val_acc'], c='red', label='val_acc')
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss / Acc / Val_acc')
plt.grid()
plt.show()

# Results 

# RobustScaler
# Epoch 754/3000
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 3s 964us/step - acc: 0.8882 - loss: 0.2742 - val_acc: 0.8837 - val_loss: 0.2848
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 669us/step - acc: 0.8859 - loss: 0.2805
# loss :  0.2804836928844452    
# acc :  0.8859
# 걸린 시간 :  2207 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 428us/step 
# acc :  0.8859

# learning_rate = 0.01
# Epoch 123/3000
# 2724/2724 [==============================] - 7s 3ms/step - loss: 0.7998 - acc: 0.7041 - val_loss: 0.7196 - val_acc: 0.7282
# 3632/3632 [==============================] - 5s 1ms/step - loss: 0.6211 - acc: 0.7460
# loss :  0.6211382150650024
# acc :  0.746
# 걸린 시간 :  761 2 초
# 3632/3632 [==============================] - 3s 745us/step
# acc :  0.746

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
# Epoch 152/2000
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - acc: 1.0000 - loss: 0.0032 - val_acc: 1.0000 - val_loss: 0.0041 - learning_rate: 6.2500e-04
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - acc: 0.9561 - loss: 0.9682 
# loss :  0.9682
# acc :  0.9561
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step

# LSTM
# Epoch 119/3000
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - acc: 0.6166 - loss: 0.9555
# Epoch 119: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 28s 10ms/step - acc: 0.6166 - loss: 0.9569 - val_acc: 0.6290 - val_loss: 0.9216 - learning_rate: 6.2500e-04
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 18s 5ms/step - acc: 0.7903 - loss: 0.5590
# loss :  0.5590441823005676
# acc :  0.7903
# 걸린 시간 :  3587 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 10s 3ms/step
# acc :  0.7903