import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random
print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1 데이터
x = np.array([1,])
y = np.array([1,])

# 모델

model = Sequential()
model.add(Dense(3,input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

print("=======================동결 ====================")
# model.trainable = False     # 동결 ★★★★★
model.trainable = True
print("=======================동결 ====================")
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=500, verbose=1)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)
print("==============================================")
print(model.weights)
print("==============================================")

# [실습]
# weights로 손계산해서 1을 찾아봅시다!!!!

# 수정필요