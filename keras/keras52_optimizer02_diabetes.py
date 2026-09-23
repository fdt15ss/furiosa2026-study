"""
03 boston
04 dacom_ddarung
05 kaggle bike
06 cancer
07 santander
08 wine
09 fetch_covtype
10 digits
"""

# R2 기준 0.62 이상
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import my_util
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
print(x_train.shape) # (331, 10)
print(y_test.shape)
model = Sequential()
model.add(Dense(20, input_dim=10))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
from keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001     # 디폴트
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = 'mse', optimizer = Adam(learning_rate=learning_rate))

from keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

path = "./_save/keras33/diabetes/"
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'k33_', date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,

)
BATCH_SIZE = 256
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 1500,
                 batch_size=BATCH_SIZE,
                 validation_split=0.125, callbacks=[es])
end_time = time.time()

#4 평가, 예측
train_time = round(end_time - start_time, 3)
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

# print("========================= hist =========================")
# print(hist)
# print("========================= hist.history =======================")
# print(hist.history)
# print("========================= hist.history['loss'] =======================")
# print(hist.history['loss'])
# print("========================= hist.history['val_loss'] =======================")
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('당뇨병 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()


my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 42,
    batch_size = BATCH_SIZE,
    # batch_size = 8,
    # history = None,
    history = hist,
    training_time = train_time,
    # training_time = 2712.563,
    test_loss = loss,
    sub_score = r2,
    train_ration = 0,
    csv_file_path="02_diabetes.csv"
)
# r2(20-10-1) : 0.5189658962934192
# r2(20-15-10-1) : 0.394818920018537
# r2(20-16-10-1) : 0.526455363169092
# r2(24-20-8-1) : 0.4379119991942728
# r2(24-12-8-1) : 0.43607822960854803
# r2(20-12-8-1) : 0.43489047037394446
# r2(20-16-10-1) : 0.42782492506749203
# r2(MinMaxScaler 적용) : 0.44209446584642864
# r2(StandardScaler 적용) : 0.4417559147428828
# r2(MaxAbsScaler 적용) : 0.43739345360832504
# r2(RobustScaler 적용) : 0.4139224277181687
# r2(save) : 0.4269347190916365
# r2(드롭아웃) : 0.41016588289174527