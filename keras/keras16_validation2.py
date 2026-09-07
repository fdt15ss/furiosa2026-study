from keras.layers import Dense
from keras.models import Sequential
import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 실습 8개, 4개, 4개 잘라봅시다!!
x_train = x[:8]
y_train = y[:8]

x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]

print(x_train.shape, y_train.shape)   #(8,) (8,)
print(x_val.shape, y_val.shape)   #(4,) (4,)
print(x_test.shape, y_test.shape)   #(4,) (4,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 100, batch_size=32, validation_data=(x_val, y_val))

# 4. 평가, 예측
model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
