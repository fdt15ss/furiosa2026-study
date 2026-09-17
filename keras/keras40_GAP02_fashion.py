import time

import numpy as np
import pandas as pd

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping



#1. 데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# import matplotlib.pyplot as plt

# plt.imshow(x_train[4342],"gray")
# plt.show()

# 스케일링 2(MaxAbs)
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=x_test[0].shape, activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (2,2), activation="relu"))
model.add(Dropout(0.2))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

batch_size = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=1, epochs=30, batch_size=batch_size, validation_split=0.3,
                    # callbacks = [es]
                    )

train_time = time.time() - start_time


#4. 평가 예측
print('훈련 시간 :', round(train_time,3), '초')
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print('acc :', acc)



#acc 0.92
#0.9181 

# 훈련 시간 : 252.688 초
# acc : 0.9157

# 훈련 시간(Flatten-e30) : 129.38 초
# acc(Flatten-e30) : 0.9191

# 훈련 시간(GlobalAveragePooling2D-e30) : 134.894 초
# acc(GlobalAveragePooling2D-e30) : 0.912