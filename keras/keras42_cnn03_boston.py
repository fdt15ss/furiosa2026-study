#11_3 카피
from keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
(x_train, y_train),(x_val_test, y_val_test) = boston_housing.load_data()

x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.4, random_state=42)

print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 13, 1, 1)
x_val = x_val.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(16, (2,1), input_shape=x_train[0].shape, activation="relu", padding="same"))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))
# model.add(Dense(26, input_dim = 13))
# model.add(Dropout(0.1))
# model.add(Dense(18))
# model.add(Dropout(0.2))
# model.add(Dense(9))
# model.add(Dense(1))

# input1 = Input(shape=(13,))
# dense1 = Dense(26)(input1)
# drop1 = Dropout(0.1)(dense1)
# dense2 = Dense(18)(drop1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(9)(drop2)
# output1 = Dense(1)(dense3)
# model = Model(inputs = input1, outputs= output1)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam') # mean squared error(평균 제곱 오차)
# loss = 에러 = 오차 = cost
# 정확도 (accuracy)
# 주지표는 loss, R²는 회귀 보조지표
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

path = "./_save/keras31/boston/"
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'k31_', date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,

)
import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 1350,
                 validation_data=(x_val, y_val),
                #  callbacks=[es, mcp])
                 callbacks=[es]
                 )
end_time = time.time()
print("===========================================")
#4. 평가, 예측
print('총 시간 :',round(end_time - start_time, 3), '초')


# 예측값 => ŷ(y hat)
loss = model.evaluate(x_test, y_test)
print('loss(mse) : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)




mse = mean_squared_error(y_test, y_predict)
print('mse :', mse)

def RMSE(y_test, y_predict):    # RMSE함수 정의
    # 파이썬은 라인 위주의 인터프리터 언어
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

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
plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][3:], c='red', label = 'loss')
plt.plot(hist.history['val_loss'][3:], c='blue', label = 'val_loss')
plt.title('보스턴 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()


# loss(mse) :  23.67670440673828
# r2 : 0.7155741314792282
# loss(mse)(MinMax스케일러 적용) :  26.679344177246094
# r2(MinMax스케일러 적용) : 0.582173716878061
# loss(mse)(StandardScaler 적용) :  23.56396484375
# r2(StandardScaler 적용) : 0.6309638217009929
# r2(MaxAbsScaler 적용) : 0.6219666381903695
# r2(RobustScaler 적용) : 0.6037485074780262
# r2(save) : 0.5825371052794688
# r2(dropout) : 0.5562519890901301
# r2(함수형) : 0.5884509206181601
# 총 시간(gpu) : 6.018 초
# 총 시간(cpu) : 9.056 초
# r2(CNN) : -0.0999666957323957