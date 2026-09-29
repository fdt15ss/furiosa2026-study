import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import time

a = np.array(range(1,101))
x_predict = np.array(range(96,106)) # 101 ~ 106까지 찾자.

size = 6

# 데이터를 reshape한 후, split_x 함수로 시계열 데이터로 변환
# (N, 10, 1) -> (N, 5, 2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

a = a.reshape(-1, 2)
bbb = split_x(a, size)
print(bbb)
x = bbb[:,:size-1]
y = bbb[:,-1]
print("===================")
x_predict = x_predict.reshape(-1, 2)
x_predict = split_x(x_predict, size - 1)
print(x_predict)

model = Sequential()
model.add(LSTM(10, input_shape=(size-1, 2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))


learning_rate = 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True,

)

start_time = time.time()
model.fit(x, y, epochs=3000, batch_size=32,
          verbose=1, validation_split=0.2,
          callbacks = [es],
          )
train_time = round(time.time() - start_time, 3)

y_predict = model.predict(x_predict)
print('y_predict: ', y_predict)
print('train_time: ', train_time, 's')