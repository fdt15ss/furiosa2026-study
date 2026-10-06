# 28-5 카피
# 데이터셋 : https://www.kaggle.com/competitions/bike-sharing-demand/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_log_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#1. 데이터
path="./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.info())
# <class 'pandas.DataFrame'>
# Index: 10886 entries, 2011-01-01 00:00:00 to 2012-12-19 23:00:00
# Data columns (total 11 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   season      10886 non-null  int64  
#  1   holiday     10886 non-null  int64  
#  2   workingday  10886 non-null  int64  
#  3   weather     10886 non-null  int64  
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64  
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64  
#  9   registered  10886 non-null  int64  
#  10  count       10886 non-null  int64  
# dtypes: float64(3), int64(8)
# memory usage: 1020.6+ KB
# None
print(test_csv.info())
# <class 'pandas.DataFrame'>
# Index: 6493 entries, 2011-01-20 00:00:00 to 2012-12-31 23:00:00
# Data columns (total 8 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   season      6493 non-null   int64  
#  1   holiday     6493 non-null   int64  
#  2   workingday  6493 non-null   int64  
#  3   weather     6493 non-null   int64  
#  4   temp        6493 non-null   float64
#  5   atemp       6493 non-null   float64
#  6   humidity    6493 non-null   int64  
#  7   windspeed   6493 non-null   float64
# dtypes: float64(3), int64(5)
# memory usage: 456.5+ KB
# None
print(submission.info())
# <class 'pandas.DataFrame'>
# Index: 6493 entries, 2011-01-20 00:00:00 to 2012-12-31 23:00:00
# Data columns (total 1 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   count   6493 non-null   int64
# dtypes: int64(1)
# memory usage: 101.5+ KB
# None

print(train_csv.describe())
#              season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000

############ 결측치 확인 #################
print(train_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
# dtype: int64
print(test_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# dtype: int64

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)    # ...[10886 rows x 8 columns]
y = train_csv['count']
print(y)    # ... Name: count, Length: 10886, dtype: int64

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
model.add(LSTM(200, input_shape=(8,1)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

#3. 컴파일, 훈련
path = './_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k53_5_", date, "-", filename])

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
    patience = 100,
    restore_best_weights = True
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
hist = model.fit(x_train, y_train, epochs=5000, batch_size=64, verbose=1, validation_split=0.25, callbacks=[es, mcp, rlr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

# 제출
# y_submit = model.predict(test_csv)

# submission['count'] = y_submit
# submission.to_csv(path + "submit/" + "submit_0910_1741.csv")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.legend(loc='upper left')
plt.title('Kaggle 자전거 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.grid()
plt.show()

# Results

# MinMaxScaler + hyper parameter tuning
# Epoch 221/5000
# 205/205 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 17268.7168 - val_loss: 21571.6660
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 705us/step - loss: 21006.5293
# loss :  21006.529296875
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 843us/step
# r2 :  0.36357229948043823
# mse :  21006.52734375
# rmse :  144.9362871876812
# 203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 444us/step

# learning_rate = 0.01
# Epoch 277/5000
# 103/103 [==============================] - 0s 2ms/step - loss: 20227.7148 - val_loss: 21467.2461
# 69/69 [==============================] - 0s 1ms/step - loss: 20983.7891
# loss :  20983.7890625
# 69/69 [==============================] - 0s 675us/step
# r2 :  0.3642612099647522
# mse :  20983.7890625
# rmse :  144.8578236150882

# learning_rate = 0.0001
# Epoch 809/5000
# 103/103 [==============================] - 0s 2ms/step - loss: 20233.6152 - val_loss: 20934.0449
# 69/69 [==============================] - 0s 775us/step - loss: 20893.1113
# loss :  20893.111328125
# 69/69 [==============================] - 0s 581us/step
# r2 :  0.3670082688331604
# mse :  20893.1171875
# rmse :  144.54451628304687

# ReduceLROnPlateau
# Epoch 138: val_loss did not improve from 3637.18457
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step - loss: 8491.7764 - val_loss: 4566.0269 - learning_rate: 6.2500e-04
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 4223.6133
# loss :  4223.61328125

# LSTM
# Epoch 160/5000
#  89/103 [========================>.....] - ETA: 0s - loss: 16078.8076
# Epoch 160: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
# 103/103 [==============================] - 0s 4ms/step - loss: 15981.2920 - val_loss: 21939.9766 - lr: 6.2500e-04
# 69/69 [==============================] - 0s 1ms/step - loss: 20968.8535
# loss :  20968.853515625
# 69/69 [==============================] - 0s 1ms/step
# r2 :  0.3647138476371765
# mse :  20968.84765625
# rmse :  144.80624177241117