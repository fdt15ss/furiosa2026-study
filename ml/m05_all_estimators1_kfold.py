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
import time

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
        start_time = time.time()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        train_time = round(time.time() - start_time, 3)
        avg_acc = round(np.mean(scores), 2)
        print(f'{name} 총 시간 :',train_time, '초')
        print('acc : ', scores, '\n평균 acc :', avg_acc)

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold )

        cross_val_predict_acc = accuracy_score(y_test, y_predict)
        print("cross_val_predict acc:", cross_val_predict_acc)
    except:
        print(name, '모델 오류')
