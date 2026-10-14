import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

print(len(model.weights)) # 30 : 13(vgg16) *2 (w,b) + 2 dense *2 (w,b)
print(len(model.trainable_weights))

print(model.weights) 