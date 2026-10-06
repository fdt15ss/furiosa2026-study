# 22 카피
# 데이터셋 : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import datetime

# 1. 내부 연산(Matrix Multiplication 등)에 사용할 스레드 수
tf.config.threading.set_intra_op_parallelism_threads(16)

# 2. 독립적인 연산들을 병렬로 처리할 스레드 수
tf.config.threading.set_inter_op_parallelism_threads(16)


#1. 데이터
path = 'c:/study/_data/kaggle_santander/'
# path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print(train_csv.shape)  # (200000, 201)

print(train_csv.isna().sum())
# target     0
# var_0      0
# var_1      0
# var_2      0
# var_3      0
#           ..
# var_195    0
# var_196    0
# var_197    0
# var_198    0
# var_199    0
# Length: 201, dtype: int64

print(train_csv.info())
# <class 'pandas.DataFrame'>
# Index: 200000 entries, train_0 to train_199999
# Columns: 201 entries, target to var_199
# dtypes: float64(200), int64(1)
# memory usage: 308.2+ MB
# None

print(test_csv.isna().sum())
print(test_csv.info())

x = train_csv.drop('target', axis=1)
print(x.shape)  # (200000, 200)
print(type(x))  # <class 'pandas.DataFrame'> 
y = train_csv['target']
print(y.shape)  # (200000,)
print(type(y))  # <class 'pandas.Series'>

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(np.unique(x, return_counts=True))
# (array([-90.2525, -83.1075, -82.2573, ...,  70.272 ,  70.8691,  74.0321], shape=(828834,)), array([1, 1, 1, ..., 1, 1, 1], shape=(828834,)))
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(200, input_shape=(200, 1)))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_7_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
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
    patience=70,
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
    callbacks=[es, mcp, rlr],
    validation_split=0.25
)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)

acc_score = accuracy_score(y_test, np.round(y_pred))
print("acc_score : ", acc_score)

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Acc/Val_acc/Loss/Val_loss')
plt.legend(loc='upper right')
plt.title('Kaggle Santander Results')
plt.grid()
plt.show()

# submit = model.predict(test_csv)
# submit = np.round(submit)   # 이진분류이므로 라운드 처리
# submission_csv['target'] = submit
# submission_csv.to_csv(path + "submit/" + "0908_1642.csv")

# Results

# MinMaxScaler
# Epoch 41/3000
# 3750/3750 ━━━━━━━━━━━━━━━━━━━━ 9s 2ms/step - acc: 0.9145 - loss: 0.2322 - val_acc: 0.9147 - val_loss: 0.2351
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 918us/step - acc: 0.9128 - loss: 0.2336
# loss :  [0.2335789054632187, 0.9127500057220459]
# 걸린 시간 :  387.88 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 744us/step
# acc_score :  0.91275
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 5s 758us/step 

# learning_rate = 0.01
# Epoch 109/3000
# 938/938 [==============================] - 2s 3ms/step - loss: 0.3251 - acc: 0.9000 - val_loss: 0.3296 - val_acc: 0.8980
# 1250/1250 [==============================] - 2s 1ms/step - loss: 0.3262 - acc: 0.8995
# loss :  [0.3261882960796356, 0.8995000123977661]
# 걸린 시간 :  255.33 초
# 1250/1250 [==============================] - 1s 737us/step
# acc_score :  0.8995

# learning_rate = 0.0001
# Epoch 84/3000
# 938/938 [==============================] - 2s 2ms/step - loss: 0.1705 - acc: 0.9379 - val_loss: 0.2666 - val_acc: 0.9089
# 1250/1250 [==============================] - 2s 1ms/step - loss: 0.2336 - acc: 0.9132
# loss :  [0.23363779485225677, 0.9131500124931335]
# 걸린 시간 :  190.48 초
# 1250/1250 [==============================] - 1s 702us/step
# acc_score :  0.91315

# ReduceLROnPlateau
# Epoch 138: val_loss did not improve from 3637.18457
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 8491.7764 - val_loss: 4566.0269 - learning_rate: 6.2500e-04
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 4223.6133
# loss :  4223.61328125

# LSTM
# Epoch 128/3000
# 938/938 [==============================] - 18s 19ms/step - loss: 0.3250 - acc: 0.9000 - val_loss: 0.3295 - val_acc: 0.8980 - lr: 1.5625e-04
# 1250/1250 [==============================] - 9s 7ms/step - loss: 0.3262 - acc: 0.8995
# loss :  [0.3261912167072296, 0.8995000123977661]
# 걸린 시간 :  2181.54 초
# 1250/1250 [==============================] - 8s 6ms/step
# acc_score :  0.8995