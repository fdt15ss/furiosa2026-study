#11_3 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
(x_train, y_train),(x_test, y_test) = boston_housing.load_data()

print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

#2. 모델구성
model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(18))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1500)

print("===========================================")
#4. 평가, 예측
# 예측값 => ŷ(y hat)
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
