import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 80을 맞춰보아요.

# split_x로 전처리해놓고,

#2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True))
model.add(LSTM(8, return_sequences=True))
model.add(LSTM(6))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()
# exit()
#3. 컴파일, 훈련
learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

es = EarlyStopping(monitor='val_loss',
                   patience = 100,
                   restore_best_weights=True,)

model.fit(x, y, epochs=2000, callbacks= [es], validation_split=0.2)

x_predict = x_predict.reshape(1,3)
print(x_predict)
y_predict = model.predict(x_predict)
print('y_predict :', y_predict)