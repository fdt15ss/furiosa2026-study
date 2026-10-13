import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
#2. 모델
# model = VGG16()
######## 디폴트 ##########
# model= VGG16(
#     weights='imagenet',
#     include_top=True,
#     input_shape=(224, 224, 3),
# )

# model.summary()

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ predictions (Dense)                  │ (None, 1000)                │       4,097,000 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 138,357,544 (527.79 MB)
#  Trainable params: 138,357,544 (527.79 MB)
#  Non-trainable params: 0 (0.00 B)

model= VGG16(
    # weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3),
)

model.summary()

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)             │ (None, 100, 100, 3)         │               0 │

# │ block5_pool (MaxPooling2D)           │ (None, 3, 3, 512)           │               0 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 14,714,688 (56.13 MB)
#  Trainable params: 14,714,688 (56.13 MB)
#  Non-trainable params: 0 (0.00 B)

############# include_top=False ##################
#1. FC layer 없어짐( 직접 아래에 fc layer를 명시 해주면 됨)
#2. input_shape를 원하는 데이터 shape로 변경