# acc : 0.95
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


#1. 데이터

datasets = load_wine()
print(datasets)
x = datasets.data
y = datasets.target

print('x.shape :', x.shape)
print(np.unique(y, return_counts=True))

y = to_categorical(y)
# print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=333, stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
from keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001     # 디폴트
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = 'categorical_crossentropy',
              optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(
    monitor="val_loss",
    patience = 20,
    mode='auto',
    restore_best_weights=True,
)


import datetime
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

path = "./_save/keras31/wine/"
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'k31_', date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,

)
BATCH_SIZE = 256
import time

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2000, validation_split=0.2,
                 batch_size=BATCH_SIZE,
                #  callbacks=[es, mcp],
                 callbacks=[es],
                 )
train_time = round(time.time() - start_time, 3)


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)
import my_util

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
    test_loss = result[0],
    sub_score = result[1],
    train_ration = 0,
    csv_file_path="08_wine.csv"
)

# loss : 0.17151986062526703
# acc : 0.9444444179534912
# acc : 0.9444444444444444

# loss : 0.14503976702690125
# acc : 0.9444444179534912
# acc : 0.9444444444444444

# loss(MinMaxScaler 적용) : 0.05554190278053284
# acc(MinMaxScaler 적용) : 0.9722222089767456
# acc(MinMaxScaler 적용) : 0.9722222222222222

# loss(MinMaxScaler transform 문제없이 적용) : 0.31127727031707764
# acc(MinMaxScaler transform 문제없이 적용) : 0.9166666865348816
# acc(MinMaxScaler transform 문제없이 적용) : 0.9166666666666666

# loss(StandardScaler 적용) : 0.17746201157569885
# acc(StandardScaler 적용) : 0.9722222089767456
# acc(StandardScaler 적용) : 0.9722222222222222

# loss(restore_best_weights=False) : 0.06486202031373978
# acc(restore_best_weights=False) : 0.9722222089767456
# acc(restore_best_weights=False) : 0.9722222222222222

# loss(MaxAbsScaler 적용) : 0.04019903391599655
# acc(MaxAbsScaler 적용) : 0.9722222089767456
# acc(MaxAbsScaler 적용) : 0.9722222222222222

# loss(save) : 0.11191385239362717
# acc(save) : 0.9444444179534912
# acc(save) : 0.9444444444444444

# loss : 0.46733182668685913
# acc : 0.9444444179534912
# acc : 0.9444444444444444