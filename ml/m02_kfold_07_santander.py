# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import my_util
from sklearn.ensemble import RandomForestClassifier

# path = 'C:/study/_data/kaggle_santander'
path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print(train_csv.shape)      #(200000, 201)
print(test_csv.shape)       #(200000, 200)
print(submission_csv.shape) #(200000, 1)

print(train_csv.info())
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([179902,  20098]))

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

scaler = StandardScaler()
x = scaler.fit_transform(x)

#2. 모델
model = RandomForestClassifier()

#3.4. 훈련, 평가
start_time = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
train_time = round(time.time() - start_time, 3)

avg_acc = round(np.mean(scores), 3)
print('총 시간 :',train_time, '초')
print('acc : ', scores, '\n평균 acc :', avg_acc)


my_util.record_model_csv(
    model = model,
    data_shape = x.shape,
    random_num = 42,
    batch_size = 0,
    # batch_size = 8,
    # history = None,
    history = None,
    training_time = train_time,
    # training_time = 2712.563,
    test_loss = 0,
    sub_score = avg_acc,
    train_ration = 0,
    csv_file_path="07_santander.csv"
)