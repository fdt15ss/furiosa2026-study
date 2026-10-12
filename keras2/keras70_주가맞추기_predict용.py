# 삼성전자 주가맞추기
# 앙상블 쓸 것
# LSTM Conv1D 각각 쓸것

import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D, Input, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime

path = '_data/samsung/'
samsung_csv = pd.read_csv(path + '삼성전자261012.csv', encoding='cp949', index_col=0)
# print(samsung_csv)

amore_csv = pd.read_csv(path + '아모레261012.csv', encoding='cp949', index_col=0)
# print(amore_csv)

# print(amore_csv.iloc[1124])
x_a_data = amore_csv.iloc[1:1125, [0,1,2,3,5,6,7,8]]
# print(x_a_data)
x_a_pred_data = amore_csv.iloc[0:5, [0,1,2,3,5,6,7,8]]

hynix_csv = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', index_col=0)
x_h_data = hynix_csv.iloc[1:1125, [0,1,2,3,5,6,7,8]]
# print(x_h_data)
x_h_pred_data = hynix_csv.iloc[0:5, [0,1,2,3,5,6,7,8]]


x_a_data = x_a_data.replace(',', '', regex=True).astype(float)
x_a_pred_data = x_a_pred_data.replace(',', '', regex=True).astype(float)

x_h_data = x_h_data.replace(',', '', regex=True).astype(float)
x_h_pred_data = x_h_pred_data.replace(',', '', regex=True).astype(float)


y = samsung_csv[:1124]['시가']
y = y.replace(',', '', regex=True).astype(float)
# print(y.head())


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x_a = split_x(x_a_data, 5)
x_h = split_x(x_h_data, 5)
x_a_pred = split_x(x_a_pred_data, 5)
x_h_pred = split_x(x_h_pred_data, 5)

x_a_train, x_a_test, x_h_train, x_h_test, y_train, y_test = train_test_split(x_a, x_h, y,
                                                                             train_size=0.8,
                                                                             random_state=42,
                                                                             shuffle=False)

scaler_a = StandardScaler()
scaler_h = StandardScaler()
print(x_a_train.shape)
x_a_train = scaler_a.fit_transform(x_a_train.reshape(-1, 5*8)).reshape(-1, 5, 8)
x_h_train = scaler_h.fit_transform(x_h_train.reshape(-1, 5*8)).reshape(-1, 5, 8)
x_a_test = scaler_a.transform(x_a_test.reshape(-1, 5*8)).reshape(-1, 5, 8)
x_h_test = scaler_h.transform(x_h_test.reshape(-1, 5*8)).reshape(-1, 5, 8)
x_a_pred = scaler_a.transform(x_a_pred.reshape(-1, 5*8)).reshape(-1, 5, 8)
x_h_pred = scaler_a.transform(x_h_pred.reshape(-1, 5*8)).reshape(-1, 5, 8)

model_path = './_save/keras70/'
model = load_model(model_path+'k70_261012_1441-.keras')

#4. 평가, 예측

y_pred = model.predict([x_a_pred, x_h_pred])
print(f'2026년 10월 13일의 삼성전자 예측 시가는 약{round(y_pred[0][0])}원입니다.')