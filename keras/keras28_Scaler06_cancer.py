import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer # 유방암 관련 데이터셋 불러오기
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

# 2개 이상은 리스트, 키 밸류는 딕셔너리
# x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape, y.shape)     # (569, 30) (569,)
print(type(x)) # <class 'numpy.ndarray'>

print(y)
print(np.unique(y)) # [0 1] -> 이진분류
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))
# 0과 1의 개수가 몇개인지 찾아보기. - pandas
print(pd.DataFrame(y).value_counts)
# 1 357
# 2 212
print("=============================================")
print(pd.Series(y).value_counts)
# 1 357
# 2 212

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=337,
    stratify=y, # 분류 모델의 경우 넣어준다. 성능은 더 좋아질 수도 있다는 정도
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([159, 239]))
# (array([0, 1]), array([148, 250])) stratify = y 넣었을 때
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([ 53, 118]))
# (array([0, 1]), array([ 64, 107])) stratify = y 넣었을 때

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer= 'adam',
            #   metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience= 10,
    restore_best_weights=True,

)

start_time = time.time()
model.fit(x_train, y_train, epochs= 1000, batch_size=32,
          verbose=1,
          callbacks=[es],
          validation_split=0.3,
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)



y_pred = model.predict(x_test)
print(y_pred[:10])
# 예측 결과에 따라 0과 1로 나눔
# y_pred = np.where(y_pred > 0.5, 1, 0) # prediction 결과를 0.5 기준으로 0 또는 1로 변환
# print(y_pred[:10])

y_pred = np.round(y_pred) # prediction 결과를 반올림하여 0 또는 1로 변환
print(y_pred[:10])


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print('acc_score :', acc_score)

# acc_score : 0.9415204678362573
# acc_score(MinMaxScaler 적용) : 0.9532163742690059