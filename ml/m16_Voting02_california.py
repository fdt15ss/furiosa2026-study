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
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()


model = VotingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat), ('LG', lg)],
)

# model = DecisionTreeRegressor()
# model = BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=100,
#     random_state=42,
#     n_jobs = -1,
#     max_samples=0.8,
#     # bootstrap=True, # 디폴트    # 중복 허용
#     bootstrap=False,          # 중복 허용x
# )

# model = RandomForestRegressor()

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
train_time = round(time.time() - start_time, 3)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

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
    sub_score = r2,
    train_ration = 0,
    csv_file_path="01_california.csv"
)

