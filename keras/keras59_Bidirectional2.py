import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 80을 맞춰보아요.
x = x.reshape(-1,3,1)
x_predict = x_predict.reshape(-1,3,1)


#[실습] 맹그러봐!!!
# Bidirectional로 리폼
model = Sequential()
model.add(Bidirectional(GRU(10), input_shape=(3,1)))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

es = EarlyStopping(monitor='loss',
                   patience=50,
                   mode='min')

learning_rate = 0.001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(x, y, epochs=50000,
          batch_size=4,
          validation_split=0.0,
          callbacks=[es])
y_predict = model.predict(x_predict)
print('y_predict :', y_predict)

