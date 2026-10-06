# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# 맹그러봐

import pandas as pd
import numpy as np
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # 메모리 모으기, 텐서플로 쓰기 전에 써야함.
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, root_mean_squared_error
import my_util

path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col = 0)

print(datasets.shape) # (420551, 14)

y_cor = datasets[-144:]['T (degC)'] # 예측치 정답 데이터
print(y_cor.shape)  # (144,)

# x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
# #1
# x_data = datasets[:-288].drop(['T (degC)'], axis=1)
# y_data = datasets[144: -144]['T (degC)']


# print(x_data.shape) # (420263, 13)
# print(y_data.shape) # (420263,)

# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         aaa.append(subset)
#     return np.array(aaa)

npy_path = '_data\\kaggle_jena_npy\\'
# x = split_x(x_data, 144)
# y = split_x(y_data, 144)

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42,)


# np.save(npy_path + 'keras58_kaggle_jena1_x_train.npy', x_train)
# np.save(npy_path + 'keras58_kaggle_jena1_x_test.npy', x_test)
# np.save(npy_path + 'keras58_kaggle_jena1_y_train.npy', y_train)
# np.save(npy_path + 'keras58_kaggle_jena1_y_test.npy', y_test)
# x_predict = split_x(x_predict, 144)
# np.save(npy_path + 'keras58_kaggle_jena1_x_predict.npy', x_predict)
# exit()
x_train = np.load(npy_path + 'keras58_kaggle_jena1_x_train.npy')
x_test = np.load(npy_path + 'keras58_kaggle_jena1_x_test.npy')
y_train = np.load(npy_path + 'keras58_kaggle_jena1_y_train.npy') 
y_test = np.load(npy_path + 'keras58_kaggle_jena1_y_test.npy') 

scaler = MinMaxScaler()
print(x_train.shape)
x_train = scaler.fit_transform(x_train.reshape(-1, 144*13)).reshape(-1, 144, 13)
x_test = scaler.transform(x_test.reshape(-1, 144*13)).reshape(-1, 144, 13)
x_predict = np.load(npy_path + 'keras58_kaggle_jena1_x_predict.npy')
x_predict = scaler.transform(x_predict.reshape(-1, 144*13)).reshape(-1, 144, 13)
# print(y.info())
print("=====================")
# print(x.info())
# exit()

# 예나를 CNN으로 만드세요
# 맹그러봐.
# print(x_train.shape)
# print("=====================")

# x_train = x_train.reshape(-1, 12, 12, 13)
# x_test = x_test.reshape(-1, 12, 12, 13)
# x_predict = x_predict.reshape(-1, 12, 12, 13)
# print(x_train.shape)
# exit()
model = Sequential()
# model.add(LSTM(32, input_shape=(144, 13), return_sequences=False))
model.add(Conv1D(16, 2, input_shape=x_train[0].shape, activation='relu'))
model.add(Conv1D(16, 2, activation='relu'))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Conv1D(64, 2, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Dropout(0.25))
# model.add(Conv1D(32, 2, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Dropout(0.25))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Dropout(0.25))
# model.add(Flatten())
model.add(GlobalAveragePooling1D())
# model.add(Conv2D(16, (3,3), input_shape=x_train[0].shape, activation="relu", padding="same"))
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())
# model.add(LSTM(32, input_shape=(144, 13), return_sequences=True))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(192))
model.add(Dense(144))

model.summary()
# exit()

learning_rate = 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,

)
import datetime
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

save_path = "./_save/kaggle_jena/"
# filename = 'ep{epoch:04d}-vl{val_loss:.4f}.keras'
filename = '.keras'
filepath = "".join([save_path, 'keras67_', date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,

)
BATCH_SIZE = 1024
start_time = time.time()
history = model.fit(x_train, y_train, epochs=3000,
          batch_size=BATCH_SIZE,
          verbose=1, validation_split=0.2,
        #   callbacks = [es],
          callbacks = [es,mcp],
          )
train_time = round(time.time() - start_time, 3)

#4 평가, 예측

loss = model.evaluate(x_test, y_test)
y_pred_test = model.predict(x_test)
r2 = r2_score(y_test, y_pred_test)
y_predict = model.predict(x_predict)

rmse = root_mean_squared_error(y_test, y_pred_test)
print("RMSE : ", rmse)

print('y_predict: ', y_predict)
print('train_time: ', train_time, 's')

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 42,
    batch_size = BATCH_SIZE,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = rmse,
    train_ration = 0,
    csv_file_path="00_kaggle_jena.csv"
)
