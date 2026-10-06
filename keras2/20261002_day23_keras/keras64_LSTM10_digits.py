#28-10 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import time

#1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = pd.get_dummies(y, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(256, input_shape=(64,1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 , 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_10_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)
es = EarlyStopping(
    monitor = 'val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=8,
    verbose=1,
    callbacks=[es, mcp, rlr],
    validation_split=0.25
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1]))

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", acc)

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='orange', label='accuracy')
plt.plot(hist.history['val_acc'], c='red', label='val_accuracy')
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Acc / Val_acc / Loss / Val_loss')
plt.legend(loc='lower center')
plt.grid()
plt.show()

# Results

# MaxAbsScaler
# Epoch 121/190000
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 1.0000 - loss: 0.0000e+00 - val_acc: 0.9889 - val_loss: 0.2497
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9833 - loss: 0.0661  
# loss :  0.0661407932639122
# acc :  1
# 걸린 시간 :  32.03 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc :  0.9833333333333333

# learning_rate = 0.01
# Epoch 101/1000
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.1040 - loss: 2.3044 - val_acc: 0.0500 - val_loss: 2.3154
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.1806 - loss: 2.2774  
# loss :  2.2773914337158203
# acc :  0
# 걸린 시간 :  33.8 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc :  0.18055555555555555

# learning_rate = 0.0001
# Epoch 129/1000
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 1.0000 - loss: 1.5197e-07 - val_acc: 0.9944 - val_loss: 0.0228
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9722 - loss: 0.1051  
# loss :  0.10510821640491486
# acc :  1
# 걸린 시간 :  40.08 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc :  0.9722222222222222

# ReduceLROnPlateau
# Epoch 152/2000
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - acc: 1.0000 - loss: 0.0032 - val_acc: 1.0000 - val_loss: 0.0041 - learning_rate: 6.2500e-04
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - acc: 0.9561 - loss: 0.9682 
# loss :  0.9682
# acc :  0.9561
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step

# LSTM
# Epoch 101: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 1s 10ms/step - acc: 0.1123 - loss: 2.3008 - val_acc: 0.0500 - val_loss: 2.3178 - learning_rate: 6.2500e-04
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - acc: 0.1000 - loss: 2.3050  
# loss :  2.304964065551758
# acc :  0
# 걸린 시간 :  159.73 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
# acc :  0.1