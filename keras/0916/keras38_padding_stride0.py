import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten

#2. 모델

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape = (10,10,1),
                padding="same", #10,10,10 빈 가장자리를 0으로 채움,
                strides=(2,1) #(5, 10, 10)
                 ))
model.add(Conv2D(9,(3,3),
                padding="valid", #디폴트 valid(8,8,9)
                strides=1
                 ))


model.summary()