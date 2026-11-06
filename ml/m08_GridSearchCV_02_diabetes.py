# 디아벳!!

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333, train_size=0.8
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
parameters = [
    {'n_estimators': [100,500, 1000], 'max_depth' :[6,8,10,12],
     "learning_rate" : [0.1, 0.01, 0.001]}, # 18 
    {'max_depth' : [6,8,10,12], 'learning_rate':[0.1,0.01,0.001]}, # 12
    {'n_estimators': [100,500, 1000],
     'min_child_weight': [2,3,5,10], 'learning_rate' : [0.1,0.01,0.001]} # 12
]   #42

#2. 모델
xgb = XGBRegressor()
model = GridSearchCV(xgb, parameters, cv=kfold,
             verbose=1,
             n_jobs=1,)

#3. 훈련
start = time.perf_counter()
model.fit(x_train,y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

#4. 평가, 예측
print("model.score :", model.score(x_test, y_test))
print("best_score :", model.best_score_)

print("걸린 시간 :", round(end - start, 2), '초')

# 최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              feature_weights=None, gamma=None, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.01, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
#              max_leaves=None, min_child_weight=10, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=500,
#              n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 :  {'learning_rate': 0.01, 'min_child_weight': 10, 'n_estimators': 500}
# model.score : 0.33383265423775843
# best_score : 0.39762802815873205
# 걸린 시간 : 105.55 초