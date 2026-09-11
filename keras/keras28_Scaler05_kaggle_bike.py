# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + '/train.csv', index_col=0)
print(train_csv)    # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + '/test.csv', index_col=0)
print(test_csv)    # [6493 rows x 8 columns]
submission = pd.read_csv(path + '/sampleSubmission.csv', index_col=0)
print(submission) # [6493 rows x 1 columns]

print(train_csv.shape)    # (10886, 11)
print(test_csv.shape)     # (6493, 8)
print(submission.shape)  # (6493, 1)

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())
########### 결측치 확인 ##############
print(train_csv.isna().sum())
print(test_csv.isnull().sum())
############ x,y 분리 ################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x) # [10886 rows x 8 columns]
y = train_csv['count']
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print(test_csv[:10])
test_csv = scaler.transform(test_csv)
# print(test_csv[:10])

#2. 모델구성

model = Sequential()
model.add(Dense(96, activation = 'relu', input_dim = 8))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
EPOCHS = 1500
BATCH_SIZE = 32
model.compile(loss = 'mse', optimizer = 'adam')

from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs = EPOCHS, batch_size=BATCH_SIZE, validation_split=0.125, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_submit = model.predict(test_csv)
# print(y_submit)
submission['count'] = y_submit

# print(submission)
from datetime import datetime
submission.to_csv(path + f'submit/submission_{datetime.now().strftime("%Y%m%d%H%M%S")}_e{EPOCHS}_b{BATCH_SIZE}.csv')


# print("========================== hist =========================")
# print(hist)
# print("========================== hist.history =========================")
# print(hist.history)
# print("========================== loss =========================")
# print(hist.history['loss'])
# print("========================== val_loss =========================")
# print(hist.history['val_loss'])
# print("=============================================================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,6))
plt.title('캐글 바이크 loss')
plt.plot(hist.history['loss'], c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

# loss(얼리스타핑) :  22206.787109375
# loss(MinMaxScaler 적용) :  21159.58984375
# loss(StandardScaler 적용) :  21192.591796875
# loss :  21744.703125
# loss(MaxAbsScaler 적용 - 초기값이 많이 튀었던 모양) :  33055.34375
# loss :  20985.083984375
# loss(RobustScaler 적용) :  21496.267578125