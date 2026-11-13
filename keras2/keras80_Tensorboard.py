import time

import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping


#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()


# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.


x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3),input_shape=x_test[0].shape)) # (26, 26, 64)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation="relu")) #(22, 22, 16)

model.add(Flatten()) #(None, 6400)
# model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()
# exit()

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

from keras.callbacks import TensorBoard

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 50,
    verbose=1,
    restore_best_weights= True, 
)

tb = TensorBoard(log_dir='c:/study/_save/tensorboard/_keras80/',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True,)

batch_size = 64
start_time = time.time()

history = model.fit(x_train,y_train,
                    verbose=2,
                    epochs=200,
                    batch_size=batch_size,
                    validation_split=0.3,
                    callbacks = [es, tb])

train_time = time.time() - start_time


#4. 평가 예측
print('훈련 시간 : ', round(train_time, 3), '초')

loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(acc)
# 훈련 시간 :  146.9895269870758
# 0.9879

# 커맨드창에서 경로 따라 들어간뒤.
# tensorboard --logdir=.

# 경로 안따라 간다면?
# tensorboard --logdir=c:/study/_save/tensorboard/

# pip install setuptools==68.2.2