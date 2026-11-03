# m02_kfold_00_iris.py 카피

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.discovery import all_estimators
import sklearn as sk
import warnings

warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333,train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
all = all_estimators(type_filter='classifier')

print('all models :', all)
print('sk version : ', sk.__version__)
print('모델의 개수 :', len(all))
# sk version :  1.9.1
# 모델의 개수 : 44

for name, model in all:
    # print(model)
    try:
        model = model()
    
        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, '모델 오류')
