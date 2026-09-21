"""
개고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이 npy데이터 사용
내 사진도 npy 불러와서 predict만 하면 되것지요. /255
끗!!!
"""

import numpy as np
from keras.models import load_model

path = "./_data/kaggle_cat_dog_npy/"
npy_path = f"{path}keras48_me.npy"
# npy_path = f"{path}keras48_dog.npy"
# npy_path = f"{path}keras48_cat.npy"

x_test = np.load(npy_path) / 255.

path = "./_save/cat_dog/"

model_path = f'{path}20260918_174700_0067-0.3918.keras'

model = load_model(model_path)


y_predict = model.predict(x_test)

print('y_predict :', y_predict) # y_predict : [[0.7586833]]