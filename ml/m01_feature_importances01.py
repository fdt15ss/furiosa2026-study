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
model1 = DecisionTreeClassifier(random_state = random_state)
model2 = RandomForestClassifier(random_state = random_state)
model3 = GradientBoostingClassifier(random_state = random_state)
model4 = XGBClassifier(random_state = random_state)

models = [model1, model2, model3, model4]



for model in models:
    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print("==================", model.__class__.__name__, "=================")

    #4. 평가, 예측
    score = model.score(x_test, y_test)
    print("acc : ",  score)
    print(model.feature_importances_) # feature engineering의 일환
