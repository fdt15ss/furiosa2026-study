import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import time


a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]],
              ).T

print(a.shape)  # (10, 2)

# 칠판처럼 짤라 보아요!!!!
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)
def split_xy(dataset, size):
    aaa = []
    bbb = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size) -1]
        aaa.append(subset)
        bbb.append(dataset[i+size - 1])
    return np.array(aaa), np.array(bbb)

# size = 5
# bbb = split_x(a, size)
# print(bbb.shape)        # (6, 5, 2)
# x = bbb[:,:-1]
# print(x)
# y = bbb[:, -1, 1]
# y = bbb[:, -1, -1]
# print("================")
# print(y)

# exit()
# x = split_x(a, 3)
x, y = split_xy(a, 5)
print(x.shape)  # (6, 4, 2)
print(y.shape)  # (6, 2)
print(x)
print()
print(y)
# exit()
model = Sequential()
model.add(LSTM(10, input_shape=(4, 2)))
model.add(Dense(6, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))

learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

es = EarlyStopping(monitor='val_loss',
                   patience=50,
                   restore_best_weights=True)
start_time = time.time()
model.fit(x, y, epochs=300,
          batch_size=8,
          callbacks=[es],
          validation_split=0.2)
train_time = time.time() - start_time
print('train_time :', train_time)

x_predict = np.array([[7,3],[8,2],[9,1],[10,0]])
x_predict = x_predict.reshape(-1,4,2)
y_predict = model.predict(x_predict)
print('y_predict[0][0] :', y_predict[0][0])
