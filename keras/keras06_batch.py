from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(15))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #로스와 옵티마이저는 일단 이걸 디폴트로
model.fit(x, y, epochs = 100, batch_size = 2)
# 90000 * 1 보단 30000 * 3 잘라서 훈련시키는 게 성능 더 좋음
# 배치사이즈가 4면 어떻게 될까? 4개, 2개 훈련
# 통배치
# 하이퍼 파라미터 튜닝 : 노드 갯수, 깊이, 배치사이즈, 에포크, 러닝레이트, 옵티마이저 등등


#4. 평가 예측
loss = model.evaluate(x, y)
print("batch_size = 2, loss : ", loss)
# result = model.predict(np.array([1,2,3,4,5,6]))
# print("result : ", result)

model.compile(loss='mse', optimizer='adam') #로스와 옵티마이저는 일단 이걸 디폴트로
model.fit(x, y, epochs = 100, batch_size = 3)
loss = model.evaluate(x, y)
print("batch_size = 3, loss : ", loss)


model.compile(loss='mse', optimizer='adam') #로스와 옵티마이저는 일단 이걸 디폴트로
model.fit(x, y, epochs = 100, batch_size = 6)
loss = model.evaluate(x, y)
print("batch_size= 6, loss : ", loss)

