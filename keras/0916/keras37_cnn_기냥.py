from keras.models import Sequential
from keras.layers import Dense, Conv2D
import keras
model = Sequential(keras.Input())
model.add(Conv2D(3, (2,2), input_shape=(5,5,1)))
#              필터 커널사이즈 input_shape (height, width, channels)
model.add(Conv2D(5,(2,2)))

model.summary()