# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온거로 컬럼을 구성해서 새로운 데이터셋을 만듬
        # : 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼을 만듦

import time
import my_util
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier

#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets['target']  # 이것도 가능.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=3444,
    stratify=y  # stratify : 계층화하다. 원데이터셋의 비율 유지하기 위함.
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2-1. 모델
xgb = XGBClassifier()
lg = LGBMClassifier(verbosity=-1)
cat = CatBoostClassifier(verbose=0)
rf = RandomForestClassifier()

model = StackingClassifier(
    estimators=[('XGB', xgb), ("LG", lg), ('CAT', cat), ('RF', rf)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5,
)


#3-2. 훈련
start_time = time.time()
model.fit(x_train, y_train)
train_time = round(time.time() - start_time, 3)

#4. 평가, 예측
y_pred = model.predict(x_test)
score_new = accuracy_score(y_test, y_pred)
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
    csv_file_path="06_cancer.csv"
)