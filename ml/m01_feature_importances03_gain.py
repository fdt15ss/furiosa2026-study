from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y= load_iris(return_X_y=True)
print(x.shape, y.shape)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = random_state,
    stratify=y
)

#2. 모델구성
model = XGBClassifier(random_state = random_state)




#3. 컴파일, 훈련
model.fit(x_train, y_train)
print("==================", model.__class__.__name__, "=================")

#4. 평가, 예측
score = model.score(x_test, y_test)
print("acc : ",  score)
print("F 스코어 :", model.feature_importances_) # feature engineering의 일환
# F 스코어 : 빈도수
gain_score = model.get_booster().get_score(importance_type='gain')
print("게인 스코어 :", gain_score)
# Gain 스코어 : 손실함수의 개서 점수. 높으면 개선 많이 됐다!!!