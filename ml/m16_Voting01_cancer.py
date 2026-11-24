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
from sklearn.ensemble import VotingClassifier, VotingRegressor

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

#2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat), ('LG', lg)],
    voting='hard', # 디폴트
    # voting='soft',
)


# model = BaggingClassifier(
#     DecisionTreeClassifier(),
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1,
#     max_samples=0.5,
#     bootstrap=True,

# )

# model = RandomForestClassifier()

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
train_time = round(time.time() - start_time, 3)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc :", acc)


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
    sub_score = acc,
    train_ration = 0,
    csv_file_path="06_cancer.csv"
)

