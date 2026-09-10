from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
# 위에 꺼 에 바이어스 1을 더한 수치를 아래 신경망 갯수 그대로에 곱한다
model.summary()