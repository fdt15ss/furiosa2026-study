from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. 컴파일, 훈련
# 최소의 로스로 최적의 웨이트를 구한다.
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs= 3000)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict(np.array([1,2,3,4,5,6,7]))
print("7의 예측값 : ", result)