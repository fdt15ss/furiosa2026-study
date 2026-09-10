#11_3 카피
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train),(x_val_test, y_val_test) = boston_housing.load_data()

x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.4, random_state=42)

print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(18))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam') # mean squared error(평균 제곱 오차)
# loss = 에러 = 오차 = cost
# 정확도 (accuracy)
# 주지표는 loss, R²는 회귀 보조지표
from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs = 1350, validation_data=(x_val, y_val), callbacks=[es])

print("===========================================")
#4. 평가, 예측
# 예측값 => ŷ(y hat)
loss = model.evaluate(x_test, y_test)
print('loss(mse) : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

# loss(mse) :  23.67670440673828
# r2 : 0.7155741314792282
# loss(mse)(스케일러 적용) :  26.679344177246094
# r2(스케일러 적용) : 0.582173716878061


mse = mean_squared_error(y_test, y_predict)
print('mse :', mse)

def RMSE(y_test, y_predict):    # RMSE함수 정의
    # 파이썬은 라인 위주의 인터프리터 언어
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print("========================== hist =========================")
print(hist)
print("========================== hist.history =========================")
print(hist.history)
print("========================== loss =========================")
print(hist.history['loss'])
print("========================== val_loss =========================")
print(hist.history['val_loss'])
print("=============================================================")

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