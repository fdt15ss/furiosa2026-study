import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],
             ])
y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)

x= x.reshape(x.shape[0], x.shape[1], 1)     # (7, 3, 1)
print(x.shape) # (7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
# model.add(SimpleRNN(16, input_shape=(3, 1)))
model.add(SimpleRNN(16, input_length=3, input_dim=1))

# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = Adam(learning_rate=0.0005))
rlr =  ReduceLROnPlateau(
    monitor='val_loss',
    patience=20,
    mode = 'auto',
    factor= 0.5,
)

es = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    mode='auto',
)

model.fit(x,y, epochs= 10000, validation_split=0.2,
          callbacks=[rlr, es])

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)