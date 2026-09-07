# R2 기준 0.62 이상
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

#2. 모델구성
print(x_train.shape) # (331, 10)
print(y_test.shape)
model = Sequential()
model.add(Dense(20, input_dim=10))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs = 500, validation_split=0.125, callbacks=[es])

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

# r2(20-10-1) : 0.5189658962934192
# r2(20-15-10-1) : 0.394818920018537
# r2(20-16-10-1) : 0.526455363169092

print("========================= hist =========================")
print(hist)
print("========================= hist.history =======================")
print(hist.history)
print("========================= hist.history['loss'] =======================")
print(hist.history['loss'])
print("========================= hist.history['val_loss'] =======================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('당뇨병 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()