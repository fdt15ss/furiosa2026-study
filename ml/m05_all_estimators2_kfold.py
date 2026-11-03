from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
import numpy as np
from sklearn.datasets import fetch_california_housing
import time
import my_util
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.utils.discovery import all_estimators


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
all = all_estimators(type_filter='regressor')

print('all models :', all)
print('모델의 개수 :', len(all))

for name, model in all:
    try:
        model = model()

        #3. 훈련
        start_time = time.time()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        train_time = round(time.time() - start_time, 3)

        #4. 평가
        
        avg_r2 = round(np.mean(scores), 2)
        print(f'{name} 총 시간 :',train_time, '초')
        print('r2 : ', scores, '\n평균 r2 :', avg_r2)

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold )

        cross_val_predict_r2 = r2_score(y_test, y_predict)
        print("cross_val_predict r2 :", cross_val_predict_r2)


        my_util.record_model_csv(
            model = model,
            data_shape = x_train.shape,
            random_num = 333,
            batch_size = 0,
            # batch_size = 8,
            # history = None,
            history = None,
            training_time = train_time,
            # training_time = 2712.563,
            test_loss = 0,
            sub_score = cross_val_predict_r2,
            train_ration = 0,
            csv_file_path="01_california.csv"
        )

    except:
        print(name, '모델 오류')