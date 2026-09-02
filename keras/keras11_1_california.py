# sklearn 데이터셋 받아지지 않을 때
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
print(x_train.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, batch_size=8, epochs= 150)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)
results = model.predict(x_test)
print(results)