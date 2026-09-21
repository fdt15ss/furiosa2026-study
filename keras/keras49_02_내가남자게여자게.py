# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

"""
keras46_03_save_npy_men_women.py
(데이터 넘파이 저장)
keras47_03_load_npy_men_women.py
(넘파이 데이터 불러와서 훈련 후, 가중치 저장)
keras49_02_내가남자게여자게.py
(넘파이와 가중치 불러와서 내 사진으로 predict)
"""

import numpy as np
from keras.models import load_model

path = "./_data/men_women_npy/"
npy_path = f"{path}keras48_me.npy"
# npy_path = f"{path}keras48_dog.npy"
# npy_path = f"{path}keras48_cat.npy"

# 255로 나눠서 스케일링 해줘야함
x_test = np.load(npy_path) / 255.

path = "./_save/men_women/"

model_path = f'{path}20260921_150906_0077-0.2640.keras'

model = load_model(model_path)


y_predict = model.predict(x_test)

print('y_predict :', y_predict) # y_predict : [[0.25699317]] (남자 : 0, 여자 : 1)