import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import time
import my_util

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor(verbosity=-1)
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

models = [xgb, lg, cat, rf]
train_list = []
test_list = []
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print(f"{class_name} r2 : {score:.4f}")

x_train_new = np.array(train_list).T
print(x_train_new.shape)
x_test_new = np.array(test_list).T
print(x_test_new.shape)

#2-2. 모델
model2 = CatBoostRegressor(verbose=0)

#3-2. 훈련
start_time = time.time()
model2.fit(x_train_new, y_train)
train_time = round(time.time() - start_time, 3)

#4. 평가, 예측
y_pred_new = model2.predict(x_test_new)
score_new = r2_score(y_test, y_pred_new)
print("스태킹 결과 :", score_new)


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
    sub_score = score_new,
    train_ration = 0,
    csv_file_path="01_california.csv"
)

