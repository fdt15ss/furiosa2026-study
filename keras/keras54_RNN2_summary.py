import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],
             ])
y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)

x= x.reshape(x.shape[0], x.shape[1], 1)     # (7, 3, 1)
print(x.shape) # (7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(SimpleRNN(2, input_shape=(3, 1))) # input_shape =(3 - timestep, 1 - feature)
# 5 * (5 + 1 + 1(bias))
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# 파라미터의 개수 = units*feature + units*bias + units*units
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 5)                 35        
                                                                 
#  dense (Dense)               (None, 7)                 42        
                                                                 
#  dense_1 (Dense)             (None, 1)                 8         
                                                                 
# =================================================================
# Total params: 85
# Trainable params: 85
# Non-trainable params: 0
# _________________________________________________________________