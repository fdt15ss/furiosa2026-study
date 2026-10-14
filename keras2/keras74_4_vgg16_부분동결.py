# keras74-3 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3) 
)
# model.trainable = False



model.layers[11].trainable=False
model.summary()

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]
results = pd.DataFrame(
    layers,columns=["Type","Name","Trainable"]
)

print(results)