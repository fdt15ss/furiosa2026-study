import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import time

a = np.array(range(1,101))
x_predict = np.array(range(96,106)) # 101 ~ 106까지 찾자.

size = 6

# loss = 0.1이하
# 결과는 
# [101, 102, 103, 104, 105, 106]의 근사치가 나오면 됨
# 106.0x ~ 105.9x 까지 인정
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
x = bbb[:,:-1]
y = bbb[:, -1]
print(x)
print("====================")
print(y)
x_predict = split_x(x_predict, size- 1)
print("====================")
print(x_predict)
# exit()


model = Sequential()
model.add(LSTM(10, input_shape=(size-1, 1)))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

learning_rate = 0.001
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
