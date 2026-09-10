# R2 기준 0.55

from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import fetch_california_housing


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# minmaxscaler
"""
MinMaxScaler

원값 - Min
-----------
Max - Min
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)
print(np.min(x),np.max(x)) # 0.0 1.0000000000000002


# exit()

x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.5, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.5, random_state=42)

#2. 모델 구성
print(x_train.shape)

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer ='adam')
hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), validation_split =0.2)

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
# r2 :  0.5977489562972026

print("========================== hist =========================")
print(hist)
print("========================== hist.history =========================")
print(hist.history)
print("========================== loss =========================")
print(hist.history['loss'])
print("========================== val_loss =========================")
print(hist.history['val_loss'])
print("=============================================================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][2:], c='red', label='loss') # y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
plt.title('캘리포니아 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 라벨 표시
plt.grid() # 격자표시 추가
plt.show()
