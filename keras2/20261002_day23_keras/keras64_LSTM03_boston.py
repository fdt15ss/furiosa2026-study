# 20-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(32, input_shape=(13,1)))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_3_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)
es = EarlyStopping(
    monitor = 'val_loss',
    mode = min,
    patience = 150,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es, mcp, rlr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6, 4))
plt.title('보스턴 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# Results
# RobustScaler + Dropout
# Epoch 452/2000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 28.4254 - val_loss: 34.3824
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 21.8889 
# loss :  21.8889217376709

# learning_rate = 0.01
# Epoch 342/2000
# 31/31 [==============================] - 0s 3ms/step - loss: 24.4551 - val_loss: 36.9898
# 4/4 [==============================] - 0s 0s/step - loss: 21.7894
# loss :  21.78935432434082

# learning_rate = 0.0001
# Epoch 588/2000
# 31/31 [==============================] - 0s 2ms/step - loss: 44.9747 - val_loss: 36.3936
# 4/4 [==============================] - 0s 0s/step - loss: 23.8932
# loss :  23.89319610595703

# ReduceLROnPlateau
# Epoch 233/2000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 28.8323 - val_loss: 31.6695 - learning_rate: 7.8125e-05
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 22.2773 

# LSTM
# Epoch 256/2000
# 31/31 [==============================] - 0s 5ms/step - loss: 15.3398 - val_loss: 19.0176 - lr: 3.9062e-05
# 4/4 [==============================] - 0s 0s/step - loss: 15.6919