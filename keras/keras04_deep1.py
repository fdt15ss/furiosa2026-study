from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(15, input_dim = 3))
model.add(Dense(4, input_dim = 15))
model.add(Dense(1, input_dim = 4))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 5000)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print("result : ", result)