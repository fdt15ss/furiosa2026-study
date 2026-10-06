from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import datetime
import time

#1, 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, # vocabulary size. 빈도수가 높은 순으로 1000개 뽑기
    # maxlen=100, # 단어 수 100개 이하로 최대 길이 제한
    test_split=0.2
)

# print(x_train)
# [list([1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 2, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 2, 2, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 2, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12])
#  list([1, 2, 699, 2, 2, 56, 2, 2, 9, 56, 2, 2, 81, 5, 2, 57, 366, 737, 132, 20, 2, 7, 2, 49, 2, 2, 2, 2, 699, 2, 8, 7, 10, 241, 16, 855, 129, 231, 783, 5, 4, 587, 2, 2, 2, 775, 7, 48, 34, 191, 44, 35, 2, 505, 17, 12])
# ...
#  list([1, 245, 273, 110, 156, 53, 272, 26, 14, 158, 26, 39, 2, 2, 14, 2, 2, 86, 32, 2, 2, 14, 19, 2, 2, 17, 12])]
# print(x_train.shape, y_train.shape)    # (8982,) (8982,)
# print(x_test.shape, y_test.shape)   # (2246,) (2246,)

# print(y_train)  # [3 4 3 ... 5 4 3]
# print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# print(type(x_train))    # <class 'numpy.ndarray'>
# print(type(x_train[0])) # <class 'list'>
# print(len(x_train[0]))  # 87

# print("뉴스기사의 최대 길이 :", max(len(i) for i in x_train))   # 뉴스기사의 최대 길이 : 2376
# print("뉴스기사의 최소 길이 :", min(len(i) for i in x_train))   # 뉴스기사의 최소 길이 : 13
# print("뉴스기사의 평균 길이 :", sum(map(len, x_train))/len(x_train))    # 뉴스기사의 평균 길이 : 145.5398574927633


# pad_sequences
x_train = pad_sequences(
    x_train,
    padding='pre',
    maxlen=146,
    truncating='post'
)

x_test = pad_sequences(
    x_test,
    padding='pre',
    maxlen=146,
    truncating='post'
)

# print(x_train)
# [[  0   0   0 ...  15  17  12]
#  [  0   0   0 ... 505  17  12]
#  [  0   0   0 ...  11  17  12]
#  ...
#  [  0   0   0 ... 407  17  12]
#  [  0   0   0 ... 364  17  12]
#  [  0   0   0 ... 113  17  12]]
# print(x_train.shape)    # (8982, 146)
# print(x_test.shape)    # (2246, 146)

# print(type(x_train))    # <class 'numpy.ndarray'>
# print(type(x_train[0])) # <class 'numpy.ndarray'>

# print("뉴스기사의 최대 길이 :", max(len(i) for i in x_train))   # 뉴스기사의 최대 길이 : 146
# print("뉴스기사의 최소 길이 :", min(len(i) for i in x_train))   # 뉴스기사의 최소 길이 : 146
# print("뉴스기사의 평균 길이 :", sum(map(len, x_train))/len(x_train))    # 뉴스기사의 평균 길이 : 146.0

# one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# print(y_train.shape, y_test.shape)  # (8982, 46) (2246, 46)

### acc 0.67 이상

#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=1000))
# model.add(LSTM(100, input_shape=(146, 1000)))  # Embedding에서 input_length가 146이고 output이 (N, 146, 1000)이므로 (N, 146, 1000) -> (146, 1000).
# input_shape에서 146으로 안 적고 다르게 적어도 에러 안 나길래 찾아보니 알아서 잘 처리해준다고 함.
model.add(Bidirectional(LSTM(100), input_shape=(146,1000)))
model.add(Dense(100))
model.add(Dense(46, activation='softmax'))
# model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

path = './_save/keras62/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path + "keras62_1_" + date + "-" + filename])

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1
)

learning_rate = 0.001

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=32,
    validation_split=0.25,
    verbose=1,
    callbacks = [es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_train, y_train)
print("loss : ", results[0])
print("acc : ", results[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print("x_test의 예측 acc : ", acc)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 107: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
# 211/211 [==============================] - 4s 17ms/step - loss: 0.0493 - acc: 0.9706 - val_loss: 1.7560 - val_acc: 0.7694 - lr: 0.0010
# Epoch 107: early stopping
# 281/281 [==============================] - 2s 8ms/step - loss: 0.5255 - acc: 0.8782
# loss :  0.5255410075187683
# acc :  0.8782008290290833
# 71/71 [==============================] - 1s 7ms/step
# x_test의 예측 acc :  0.7600178094390027
# 걸린 시간 :  401.51 초