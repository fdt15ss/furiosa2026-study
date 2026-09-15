from keras.models import Sequential
from keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape= (10,10,1,)))
model.add(Conv2D(5,(2,2)))
# 특성이 큰 건 더 크게, 작은 건 더 소멸하게

model.summary()
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 8, 8, 10)            │             100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 7, 7, 5)             │             205 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 305 (1.19 KB)
#  Trainable params: 305 (1.19 KB)
#  Non-trainable params: 0 (0.00 B)