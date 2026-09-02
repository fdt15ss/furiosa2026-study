import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

#[찾아보기] 넘파이 리스트의 슬라이싱 => 7:3 으로 나누자
# x_train = x[0:7]
x_train = x[:7]
y_train = y[:7]
# x_test = x[7:10]
x_test = x[7:]
y_test = y[7:]
print(x_test)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 4) # 배치 사이즈가 전체 데이터보다 클 경우 통배치가 돌아간다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 통상적으로 평가 loss는 훈련 loss보다 높게 나온다. 훈련 데이터에 과적합이 일어나면 평가 loss가 더 높게 나온다.
# train과 달리 최적의 weight를 업데이트하지 않기 때문에 테스트셋 evaluate는 배치를 통으로 돌린다.
print('loss : ', loss)