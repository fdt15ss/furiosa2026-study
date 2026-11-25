# 19-1 카피

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor
import time
import my_util
#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators' : 2000,
    'learning_rate' : 0.05,
    'max_depth' : 7,
    'gamma' : 1,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 0,
    'random_state' : 3377,
    # 'verbose' : 0,
}

#2. 모델
model = XGBRegressor(
    # **parameters
)
model.set_params(
    **parameters,
    early_stopping_rounds=20,
)


#3. 훈련
start_time = time.time()
model.fit(x_train, y_train,
        eval_set = [(x_test, y_test)],
        verbose = 1,
)
          
train_time = round(time.time() - start_time, 3)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

path = './_save/pickle/'
import joblib
import pickle
# joblib.dump(model, path + 'm19_joblib1_save.joblib')
pickle.dump(model, open(path + 'm20_pickle1_save.pickle', 'wb'))

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


