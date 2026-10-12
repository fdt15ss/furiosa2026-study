# 삼성전자 주가맞추기
# 앙상블 쓸 것
# LSTM Conv1D 각각 쓸것

import pandas as pd
from keras.models import Model
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

print(amore_csv.iloc[1124])
x_a_data = amore_csv.iloc[1:1125, [0,1,2,3,5,6,7,8]]
print(x_a_data)
x_a_pred_data = amore_csv.iloc[0:5, [0,1,2,3,5,6,7,8]]

hynix_csv = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', index_col=0)
x_h_data = hynix_csv.iloc[1:1125, [0,1,2,3,5,6,7,8]]
print(x_h_data)
x_h_pred_data = hynix_csv.iloc[0:5, [0,1,2,3,5,6,7,8]]


x_a_data = x_a_data.replace(',', '', regex=True).astype(float)
x_a_pred_data = x_a_pred_data.replace(',', '', regex=True).astype(float)

x_h_data = x_h_data.replace(',', '', regex=True).astype(float)
x_h_pred_data = x_h_pred_data.replace(',', '', regex=True).astype(float)


y = samsung_csv[:1124]['시가']
y = y.replace(',', '', regex=True).astype(float)
print(y.head())

# exit()
# x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
# #1
# x_data = datasets[:-288].drop(['T (degC)'], axis=1)
# y_data = datasets[144: -144]['T (degC)']

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

# 2-1. 모델
input11 = Input(shape=(5,8))
lstm11 = LSTM(10, return_sequences=True)(input11)
conv1D11 = Conv1D(16, 2, activation='relu')(lstm11)
conv1D12 = Conv1D(16, 2, activation='relu')(conv1D11)
flatten11 = Flatten()(conv1D12)
dense11 = Dense(32, activation='relu')(flatten11)
output1 = Dense(8, activation='relu')(dense11)

# 2-2. 모델
input21 = Input(shape=(5,8))
lstm21 = LSTM(10, return_sequences=True)(input21)
conv1D21 = Conv1D(16, 2, activation='relu')(lstm21)
conv1D22 = Conv1D(16, 2, activation='relu')(conv1D21)
flatten21 = Flatten()(conv1D22)
dense21 = Dense(32, activation='relu')(flatten21)
output2 = Dense(8, activation='relu')(dense21)

# 2-3. 모델 합치기
merge1 = Concatenate()([output1, output2])
merge2 = Dense(8, activation='relu')(merge1)
merge3 = Dense(4, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input11, input21], outputs= last_output)
model.summary()
# exit()
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

path = './_save/keras70/'
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
filename = '.keras'
filepath = "".join([path, "k70_", date, "-", filename])

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    save_best_only=True,
)
BATCH_SIZE = 32

history = model.fit([x_a_train, x_h_train], y_train, epochs=2000, batch_size=BATCH_SIZE,
          validation_split=0.2,
          callbacks=[es, mcp]
          )

#4. 평가, 예측
result = model.evaluate([x_a_test, x_h_test], y_test)
print('loss : ', result)

y_pred = model.predict([x_a_pred, x_h_pred])
print(y_pred)