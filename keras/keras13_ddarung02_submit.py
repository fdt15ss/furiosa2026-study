# https://dacon.io/competitions/official/235576/overview/description

import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import pandas as pd


#1. 데이터
# path = "./_data/ddarung/"             # 상대경로
# path = "c:/study/_data/ddarung/"    # 절대경로
# path = 'c:\study\_data\ddarung/'    # \s를 인식해서 에러
# path = "c:\study\_data\ddarung/"    # 슬래시 역슬래시 상관없어
path = "c:\\study\\_data\\ddarung\\"  # 됨
# path = "'c:\\study\_data\\ddarung/"   # 섞어쓰기 되지만 가급적 비권장.

train_csv = pd.read_csv(path + 'train.csv', index_col=0) # 행 = row
print(train_csv)    
# id열 포함 [1459 rows x 11 columns]
# id열 안포함 [1459 rows x 10 columns] (, index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) # [715 rows x 9 columns]

# 결측치를 NaN으로 많이 얘기함
submission = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission)   # [715 rows x 1 columns]

print(train_csv.shape)  # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission.shape) # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')
print(train_csv.info())
print(test_csv.info())

# exit()
############################# 결측치 처리 1. 삭제 ###############
train_csv = train_csv.dropna()
print(train_csv)    # [1328 rows x 10 columns]

################# train_csv를 x와 y로 분리 ######################
x = train_csv.drop(['count'], axis=1)  # 열(컬럼) 삭제
print(x)

y = train_csv['count']
print(y)
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

print('x_train.shape:', x_train.shape) # (996, 9)
print('x_test.shape:', x_test.shape) # (332, 9)
print('y_train.shape:', y_train.shape) # (996,)
print('y_test.shape:', y_test.shape) # (332,)

########################## submit 물밑 작업 #########################
print(test_csv.info())

########################## 결측치 처리 2. 평균값 넣기 ######################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())
print(test_csv.shape) # (715, 9)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
# mae (평균 절대 오차)
model.fit(x_train, y_train, epochs=500, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
rmse = root_mean_squared_error(y_test, y_predict)
print('rmse :', rmse)

###################### submission.csv 만들기 // count 컬럼에 값 넣어준다. ############################
print(submission)
y_submit = model.predict(test_csv)
submission['count'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + 'submit/' + 'submit_20260904_03.csv')
