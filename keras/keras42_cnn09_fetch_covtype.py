# acc : 0.93
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import time

#1. 데이터
datasets = fetch_covtype()
# print(datasets)
x = datasets.data # (581012, 54)
y = datasets.target

print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
y_cat = to_categorical(y) # 
y = pd.get_dummies(y, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.8,
                                                    stratify=y,
                                                    random_state = 333)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 9, 6, 1)
x_test = x_test.reshape(-1, 9, 6, 1)

print('x_train.shape :', x_train.shape)     # x_train.shape : (464809, 54)
print('x_test.shape :', x_test.shape)       # x_test.shape : (116203, 54)
print('y_train.shape :', y_train.shape)     # y_train.shape : (464809, 7)
print('y_test.shape :', y_test.shape)       # y_test.shape : (116203, 7)
print('y_cat.shape :', y_cat.shape)         # y_cat.shape : (581012, 8)
print(y_cat[:10])
print(y_test[:10])
# exit()
#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(9,6,1), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.add(Dense(256, input_dim=54, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# input1 = Input(shape=(54,))
# dense1 = Dense(256, activation='relu')(input1)
# dense2 = Dense(256, activation='relu')(dense1)
# dense3 = Dense(256, activation='relu')(dense2)
# drop1 = Dropout(0.2)(dense3)
# dense4 = Dense(128, activation='relu')(drop1)
# dense5 = Dense(128, activation='relu')(dense4)
# dense6 = Dense(128, activation='relu')(dense5)
# drop2 = Dropout(0.2)(dense6)
# dense7 = Dense(64, activation='relu')(drop2)
# dense8 = Dense(64, activation='relu')(dense7)
# drop3 = Dropout(0.2)(dense8)
# dense9 = Dense(32, activation='relu')(drop3)
# dense10 = Dense(32, activation='relu')(dense9)
# dense11 = Dense(16, activation='relu')(dense10)
# dense12 = Dense(16, activation='relu')(dense11)
# output1 = Dense(7, activation='softmax')(dense12)
# model = Model(inputs = input1, outputs = output1)
model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)


import datetime
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

path = "./_save/keras31/fetch_covtype/"
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'k31_', date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,

)
start_time = time.time()

hist = model.fit(x_train, y_train,
          epochs = 500,
        #   callbacks=[es, mcp],
          callbacks=[es, ],
          validation_split=0.2,
          batch_size=4096
          )
end_time = time.time()


#4. 평가, 예측
print('총 시간 :',round(end_time - start_time, 3), '초')

result = model.evaluate(x_test, y_test)
print(f'loss : {round(result[0], 4)}, acc: {round(result[1], 4)}')

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict)
# y_test = np.argmax(y_test)
# acc = accuracy_score(y_test, y_predict)

acc = np.mean(categorical_accuracy(y_test, y_predict))
print('acc :', acc)

# loss : 0.1365, acc: 0.9502
# acc : 0.95019925
# loss(MinMaxScaler 적용, 배치 4096) : 0.167, acc: 0.9354
# acc(MinMaxScaler 적용, 배치 4096) : 0.9353631
# loss(StandardScaler 적용, 배치 1024) : 0.1254, acc: 0.9559
# acc(StandardScaler 적용, 배치 1024) : 0.9559306

# loss(MaxAbsScaler 적용) : 0.1444, acc: 0.9449
# acc(MaxAbsScaler 적용) : 0.9448723

# loss(save) : 0.147, acc: 0.9444
# acc(save) : 0.9443646

# acc(dropout) : 0.94651604
# 총 시간(gpu) : 34.388 초
# 총 시간(cpu) : 84.689 초

# 총 시간(CNN-3060) : 210.698 초
# loss(CNN-3060) : 0.3788, acc: 0.8452
# acc(CNN-3060) : 0.8451589