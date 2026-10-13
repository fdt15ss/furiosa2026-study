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
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 모델

model = Sequential()
model.add(Dense(30,input_dim = 1))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

print("=======================동결 ====================")
model.trainable = False     # 동결 ★★★★★
# model.trainable = True
print("=======================동결 ====================")
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100, verbose=1)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

# 동결 x
# [[1.004619 ]
#  [2.0028946]
#  [3.0011704]
#  [3.9994462]
#  [4.9977217]]

# 동결 o
# [[0.09261152]
#  [0.18522304]
#  [0.27783448]
#  [0.3704461 ]
#  [0.4630577 ]]

# 수정필요