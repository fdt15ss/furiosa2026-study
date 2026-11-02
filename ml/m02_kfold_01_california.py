from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.datasets import fetch_california_housing
import time
import my_util
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

#2. 모델
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

#3.4. 훈련, 평가
start_time = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
train_time = round(time.time() - start_time, 3)

avg_r2 = round(np.mean(scores), 2)
print('총 시간 :',train_time, '초')
print('r2 : ', scores, '\n평균 r2 :', avg_r2)


my_util.record_model_csv(
    model = model,
    data_shape = x.shape,
    random_num = 0,
    batch_size = 0,
    # batch_size = 8,
    # history = None,
    history = None,
    training_time = train_time,
    # training_time = 2712.563,
    test_loss = 0,
    sub_score = avg_r2,
    train_ration = 0,
    csv_file_path="01_california.csv"
)