# 28-4 카피
# 데이터셋 - https://dacon.io/competitions/open/235576/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

#1. 데이터
path = "./_data/ddarung/"   # 상대경로
# path = "c:/study/_data/ddarung/" # 절대경로
# path = "c:\study\_data\ddarung\" # 끝에 역슬래시 + 예약어랑 겹쳐서 에러남. \s, \n, \t
# path = "c://study//_data//ddarung/" # 이것도 가능
# path = "c:\\study\\_data\\ddarung\\" # 역슬래시에서 예약어랑 겹침 방지
# path = "c:\study\\_data/ddarung\\" # 가능은 함.



train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
# id열 포함 [1459 rows x 11 columns]
# index_col=0 쓰면 id열(0번째 열)을 인덱스로 써서 실제 데이터는 한 줄 줄어듬 [1459 rows x 10 columns]
# 맨 윗줄은 컬럼명으로 처리. 데이터 포함 안 함. 맨 첫 행 제외해서 데이터만 1459 행임.

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) # ... [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)    # ... [715 rows x 1 columns]

print(train_csv.shape, test_csv.shape, submission.shape)    # (1459, 10) (715, 9) (715, 1)

print(train_csv.columns)    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='str')

print(train_csv.info())
# ..#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
# memory usage: 125.4 KB
# None
# 결측지 존재하므로 결측치 처리 필요

print(test_csv.info())
# <class 'pandas.DataFrame'>
# Index: 715 entries, 0 to 2177
# Data columns (total 9 columns):
#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    715 non-null    int64  
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64
# dtypes: float64(8), int64(1)
# memory usage: 55.9 KB
# None
# 결측지 존재하므로 결측치 처리 필요

# ----------------------결측치 처리 1. 삭제--------------------
train_csv = train_csv.dropna()
print(train_csv)    # ... [1328 rows x 10 columns]

######### train_csv를 x와 y로 분리. ########
x = train_csv.drop(['count'], axis=1)   # count 열 삭제. axis=0은 행 삭제
print(x)    # ... [1328 rows x 9 columns]

y = train_csv['count']
print(y)    # ... Name: count, Length: 1328, dtype: float64
print(y.shape)  # (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape for RNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델 구성
model = Sequential()
model.add(LSTM(180, input_shape=(9,1)))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_4_", date, "-", filename])

learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)
es = EarlyStopping(
    monitor = 'val_loss',
    mode = min,
    patience = 200,
    restore_best_weights = True
)
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=190000, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, mcp, rlr])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

"""
    python-기초
    블럭 주석 처리-" 3개

    하이퍼파라미터 튜닝
        - radom_state
        - train_size
        - 레이어 depth
        - 노트 갯수
        - epochs
        - batch_size
"""

"""
    파라미터 튜닝 시 파라미터값 적어두기. 명세.
"""
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False  # minus 기호 깨짐 방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='orange', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.legend(loc='upper right')
plt.title('따릉이 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.grid()
plt.show()

# Results
# MaxAbsScaler
# Epoch 528/190000
# 25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 182.1598 - val_loss: 1841.1351
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1783.9213
# loss :  1783.9212646484375
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# mse :  1783.921283274397
# rmse :  42.23649231735984
# 걸린 시간 :  42.45 초

# learning_rate = 0.01
# Epoch 307/190000
# 25/25 [==============================] - 0s 3ms/step - loss: 1549.5232 - val_loss: 1933.4194
# 9/9 [==============================] - 0s 2ms/step - loss: 1935.4855
# loss :  1935.4854736328125
# 9/9 [==============================] - 0s 1ms/step
# mse :  1935.4853809194217
# rmse :  43.99415166723211
# 걸린 시간 :  20.03 초

# learning_rate = 0.0001
# Epoch 3302/190000
# 25/25 [==============================] - 0s 3ms/step - loss: 1131.1506 - val_loss: 1740.9701
# 9/9 [==============================] - 0s 881us/step - loss: 1784.6056
# loss :  1784.6055908203125
# 9/9 [==============================] - 0s 961us/step
# mse :  1784.6054597734085
# rmse :  42.24459089366837
# 걸린 시간 :  229.43 초

# ReduceLROnPlateau
# Epoch 138: val_loss did not improve from 3637.18457
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 8491.7764 - val_loss: 4566.0269 - learning_rate: 6.2500e-04
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 4223.6133
# loss :  4223.61328125

# LSTM
# Epoch 267: ReduceLROnPlateau reducing learning rate to 9.765624781721272e-06.
# 25/25 [==============================] - 0s 5ms/step - loss: 286.8738 - val_loss: 1696.2826 - lr: 1.9531e-05
# 9/9 [==============================] - 0s 2ms/step - loss: 1769.2661
# loss :  1769.26611328125
# 9/9 [==============================] - 0s 2ms/step
# mse :  1769.2661410370197
# rmse :  42.06264543555267
# 걸린 시간 :  32.54 초