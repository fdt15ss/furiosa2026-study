from sklearn.datasets import load_digits
# acc : 1.0
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import time

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

ohe = OneHotEncoder(sparse_output=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)

print(f'x.shape : {x.shape}, y.shape : {y.shape}')
print(f'x.dtype : {x.dtype}, y.dtype : {y.dtype}')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8, stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 4, 2)
x_test = x_test.reshape(-1, 8, 4, 2)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(8,4,2), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
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
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# model.add(Dense(256, input_dim=64, activation='relu'))
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
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(64,))
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
# drop3 = Dropout(0.3)(dense8)
# dense9 = Dense(32, activation='relu')(drop3)
# dense10 = Dense(32, activation='relu')(dense9)
# dense11 = Dense(32, activation='relu')(dense10)
# dense12 = Dense(32, activation='relu')(dense11)
# drop4 = Dropout(0.5)(dense12)
# dense13 = Dense(16, activation='relu')(drop4)
# dense14 = Dense(16, activation='relu')(dense13)
# output1 = Dense(10, activation='softmax')(dense14)
# model = Model(inputs = input1, outputs= output1)

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   mode='auto',
                   restore_best_weights=True
                   )


import datetime
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

path = "./_save/keras31/digits/"
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
                 epochs=1000, batch_size=128,
                 validation_split=0.2,
                #  callbacks=[es, mcp])
                 callbacks=[es, ]
                 )
end_time = time.time()


#4. 평가, 예측
print('총 시간 :',round(end_time - start_time, 3), '초')

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
acc_categorical = np.mean(categorical_accuracy(y_test, y_predict))
print('acc_categorical :', acc_categorical)

# loss :  0.07446880638599396
# acc :  0.9916666746139526
# acc_categorical : 0.9916667
# loss(MinMaxScaler 적용) :  0.20266978442668915
# acc(MinMaxScaler 적용) :  0.949999988079071
# acc_categorical(MinMaxScaler 적용) : 0.95
# loss(StandardScaler 적용) :  0.18896925449371338
# acc(StandardScaler 적용) :  0.9666666388511658
# acc_categorical(StandardScaler 적용) : 0.96666664

# loss(restore_best_weights=False) :  0.3095552623271942
# acc(restore_best_weights=False) :  0.9638888835906982
# acc_categorical(restore_best_weights=False) : 0.9638889

# loss(MaxAbsScaler 적용, restore_best_weights=True) :  0.1736135333776474
# acc(MaxAbsScaler 적용, restore_best_weights=True) :  0.9611111283302307
# acc_categorical(MaxAbsScaler 적용, restore_best_weights=True) : 0.9611111

# loss(save) :  0.1458550989627838
# acc(save) :  0.9611111283302307
# acc_categorical(save) : 0.9611111

# acc_categorical(Dropout) : 0.98055553
# 총 시간(gpu) : 7.04 초
# 총 시간(cpu) : 11.916 초

# 총 시간(CNN-3060) : 11.464 초
# loss(CNN-3060) :  1.0304656028747559
# acc(CNN-3060) :  0.7138888835906982
# acc_categorical(CNN-3060) : 0.7138889