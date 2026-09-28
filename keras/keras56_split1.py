import numpy as np

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU

a = np.array(range(1, 11))
size = 6        # timestep 사이즈

print(a.shape)  # (10,)

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

x, y = split_xy(a, size)
print(x)
print(y)

model = Sequential()
model.add(GRU(10, input_shape=(5,1)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')



model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=2000, validation_split=0.2,
          callbacks=[es],
          batch_size=32)

x_predict = np.array([[7,8,9,10,11]])
x_predict = x_predict.reshape(1,5,1)
y_predict = model.predict(x_predict)
print('y_predict :', y_predict)

# y_predict(LSTM) : [[9.459713]]
# y_predict(GRU) : [[9.350899]]