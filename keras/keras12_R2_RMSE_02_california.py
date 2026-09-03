# R2 기준 0.55

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y)

#2. 모델 구성
print(x_train.shape)

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer ='adam')
model.fit(x_train, y_train, epochs=200)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# loss : 0.6413015723228455
# r2(16-8-1) :  0.5116739162297916
# r2 :  0.5006368636230876
# r2 :  0.4866390989900473
# r2(16-1) :  0.010727020275410504
# r2(16-12-8-1) :  0.5249744806146608