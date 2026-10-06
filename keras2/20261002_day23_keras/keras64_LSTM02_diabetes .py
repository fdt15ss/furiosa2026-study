# 28-2 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.25,
    random_state=42
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련

path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_2_", date, "-", filename])

# learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
learning_rate = 0.0001
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
    patience = 70,
    restore_best_weights = False
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only= True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=2400, batch_size=256, verbose=1, validation_data=(x_val, y_val), callbacks=[es, mcp, rlr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,4))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.legend(loc='upper right')
plt.title('Diabetes Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.grid()
plt.show()

# Results

# learning_rate = 0.01
# Epoch 211/2400
# 2/2 [==============================] - 0s 14ms/step - loss: 7511.0366 - val_loss: 4147.9580
# 3/3 [==============================] - 0s 1ms/step - loss: 3715.3179
# loss :  3715.31787109375

# learning_rate = 0.0001
# Epoch 1984/2400
# 2/2 [==============================] - 0s 28ms/step - loss: 9285.4912 - val_loss: 6303.0176
# 3/3 [==============================] - 0s 3ms/step - loss: 5614.4839
# loss :  5614.4838867187

# ReduceLROnPlateau
# Epoch 138: val_loss did not improve from 3637.18457
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 8491.7764 - val_loss: 4566.0269 - learning_rate: 6.2500e-04
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 4223.6133
# loss :  4223.61328125

# LSTM
# Epoch 1258/2400
# 1/2 [==============>...............] - ETA: 0s - loss: 9706.7588
# Epoch 1258: val_loss did not improve from 7774.88818
# 2/2 [==============================] - 0s 24ms/step - loss: 9864.8379 - val_loss: 7776.1069 - lr: 3.1250e-06
# 3/3 [==============================] - 0s 2ms/step - loss: 6108.0386
# loss :  6108.03857421875