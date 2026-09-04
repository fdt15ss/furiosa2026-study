# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + '/train.csv', index_col=0)
print(train_csv)    # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + '/test.csv', index_col=0)
print(test_csv)    # [6493 rows x 8 columns]
submission = pd.read_csv(path + '/sampleSubmission.csv', index_col=0)
print(submission) # [6493 rows x 1 columns]

print(train_csv.shape)    # (10886, 11)
print(test_csv.shape)     # (6493, 8)
print(submission.shape)  # (6493, 1)

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())
########### 결측치 확인 ##############
print(train_csv.isna().sum())
print(test_csv.isnull().sum())
############ x,y 분리 ################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x) # [10886 rows x 8 columns]
y = train_csv['count']
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

#2. 모델구성

model = Sequential()
model.add(Dense(96, activation = 'relu', input_dim = 8))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
EPOCHS = 500
BATCH_SIZE = 32
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = EPOCHS, batch_size=BATCH_SIZE)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_submit = model.predict(test_csv)
print(y_submit)
submission['count'] = y_submit

print(submission)
from datetime import datetime
submission.to_csv(path + f'submit/submission_{datetime.now().strftime("%Y%m%d%H%M%S")}_e{EPOCHS}_b{BATCH_SIZE}.csv')

