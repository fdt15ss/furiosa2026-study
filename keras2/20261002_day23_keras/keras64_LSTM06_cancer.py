# 21-1 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time
import matplotlib.pyplot as plt
import datetime

'''
회귀코드와의 차이
1. 요소 균형 확인 - unique
2. 훈련,테스트 계층화해서 분할 - stratify
*** 3. 레이어 마지막 activation function 고정 - sigmoid ***
*** 4. 컴파일에서 loss function 고정(이진분류에서) - binary_crossentropy ***
'''

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets['target']  # 이것도 가능.

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>
# 파이썬 기초 - type() 타입 반환

print(y)
# 0과 1의 갯수가 몇 개인지 찾아보기 - numpy
print(np.unique(y)) # [0 1]. np.unique() : 들어있는 요소들 확인
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357])). return_counts : 요소별 개수 확인

# 0과 1의 갯수가 몇 개인지 찾아보기 - pandas
print(pd.DataFrame(y).value_counts())
# 0
# 1    357
# 0    212
# Name: count, dtype: int64
print(pd.Series(y).value_counts()) # 얘도 가능. 결과는 동일.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # stratify : 계층화하다. 원데이터셋의 비율 유지하기 위함.
)

print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([170, 285]))
print(np.unique(y_test, return_counts=True))    # (array([0, 1]), array([42, 72]))

# scaler = MinMaxScaler()
scaler = StandardScaler()
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
model.add(LSTM(30, input_shape=(30, 1)))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))   # 끝은 무조건 sigmoid로.

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_6_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy', # loss는 무조건 binary_crossentropy로 고정(이진 분류에서)
              optimizer=Adam(learning_rate=learning_rate),
              # metrics=['accuracy']  # 보조지표 accuracy
              metrics=['acc'] # 이것도 가능
              )
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)
es = EarlyStopping(
    monitor='val_loss',
    mode=min,
    patience=50,
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
hist = model.fit(x_train, y_train, epochs=2000, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, mcp, rlr])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0], 4)) # loss :  0.2158
print("acc : ", round(loss[1], 4))  # acc :  0.9298

y_pred = model.predict(x_test)
# print(y_pred)   # [[2.19274860e-16] [1.00000000e+00] ... [9.97549653e-01]]. sigmoid쓰면 0, 1 사이 값 되고 반올림에서 실제 분류 처리함.

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.plot(hist.history['acc'], c='blue', label='accuracy')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.title('Breast Cancer Loss')
plt.grid()
plt.show()

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, np.round(y_pred))    # accuracy_score를 위해 y_pred round 처리
print("acc_score :", acc_score) # acc_score : 0.9210526315789473

# Results

# StandardScaler
# Epoch 64/2000
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - acc: 1.0000 - loss: 7.7650e-05 - val_acc: 0.9825 - val_loss: 0.0732
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 0.9825 - loss: 0.0623 
# loss :  0.0623
# acc :  0.9825
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
# ...
# 걸린 시간 :  5.19 초
# acc_score : 0.9824561403508771

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
# Epoch 152/2000
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - acc: 1.0000 - loss: 0.0032 - val_acc: 1.0000 - val_loss: 0.0041 - learning_rate: 6.2500e-04
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - acc: 0.9561 - loss: 0.9682 
# loss :  0.9682
# acc :  0.9561
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step

# LSTM
# Epoch 103/2000
# 11/11 [==============================] - 0s 6ms/step - loss: 0.0341 - acc: 0.9883 - val_loss: 0.3333 - val_acc: 0.9649 - lr: 0.0025
# 4/4 [==============================] - 0s 4ms/step - loss: 0.1479 - acc: 0.9298
# loss :  0.1479
# acc :  0.9298
# 4/4 [==============================] - 0s 3ms/step
# 걸린 시간 :  10.88 초
# acc_score : 0.9298245614035088