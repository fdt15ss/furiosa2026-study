# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import time

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

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

scaler = RobustScaler()
x = scaler.fit_transform(x)
# print(test_csv[:10])
test_csv = scaler.transform(test_csv)

#2. 모델
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

#3.4. 훈련, 평가
start_time = time.time()
scores = cross_val_score(model, x, y, cv = kfold)
train_time = round(time.time() - start_time, 3)

avg_r2 = round(np.mean(scores), 3)
print('총 시간 :',train_time, '초')
print('r2 : ', scores, '\n평균 r2 :', avg_r2)




import my_util
my_util.record_model_csv(
    model = model,
    data_shape = x.shape,
    random_num = 333,
    batch_size = 0,
    # batch_size = 8,
    # history = None,
    history = None,
    training_time = train_time,
    # training_time = 2712.563,
    test_loss = 0,
    sub_score = avg_r2,
    train_ration = 0,
    csv_file_path="05_kaggle_bike.csv"
)