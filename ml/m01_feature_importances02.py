from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
x, y= load_diabetes(return_X_y=True)
print(x.shape, y.shape)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = random_state,
)

#2. 모델구성
model1 = DecisionTreeRegressor(random_state = random_state)
model2 = RandomForestRegressor(random_state = random_state)
model3 = GradientBoostingRegressor(random_state = random_state)
model4 = XGBRegressor(random_state = random_state)

models = [model1, model2, model3, model4]



for model in models:
    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print("==================", model.__class__.__name__, "=================")

    #4. 평가, 예측
    score = model.score(x_test, y_test)
    print("R2 : ",  score)
    print(model.feature_importances_) # feature engineering의 일환
