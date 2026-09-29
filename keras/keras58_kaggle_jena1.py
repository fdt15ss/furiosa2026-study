# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

import pandas as pd
import numpy as np
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # 메모리 모으기, 텐서플로 쓰기 전에 써야함.
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
import my_util

path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col = 0)

print(datasets.shape) # (420551, 14)

y_cor = datasets[-144:]['wd (deg)'] # 예측치 정답 데이터
print(y_cor.shape)  # (144,)

# x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
# #1
# x_data = datasets[:-288].drop(['wd (deg)'], axis=1)
# y_data = datasets[144: -144]['wd (deg)']


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

model = Sequential()
model.add(LSTM(32, input_shape=(144, 13), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(144))


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
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([save_path, 'k31_', date, "-", filename])


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
    sub_score = r2,
    train_ration = 0,
    csv_file_path="00_kaggle_jena.csv"
)
