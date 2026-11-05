# '캘리포니아', '디아벳'
# 맹그러봐!!!

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.discovery import all_estimators
import sklearn as sk
import warnings
import time
warnings.filterwarnings('ignore')

#1. 데이터
california = (fetch_california_housing().data, fetch_california_housing().target)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['캘리포니아', '디아벳']

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
            x,y, shuffle=True, random_state=333,train_size=0.8,
        )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #2. 모델
    all = all_estimators(type_filter = 'regressor')
    print("=====================", data_name[index], "=========================")
    model_name = []
    model_r2 = []

    for name, model in all:
        try:
            model = model()
                    
            #3. 훈련
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            
            #4. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold )

            r2 = np.round(r2_score(y_test, y_pred), 4)
            model_name.append(name)
            model_r2.append(r2)
        except:
            continue

    model_name = np.array(model_name)
    model_r2 = np.array(model_r2)
    max_index = np.argmax(model_r2)
    print("최고 모델 :", model_name[max_index], 'R2 : ', model_r2[max_index])
end = time.time()

print("걸린 시간 :", round(end - start, 2), '초')