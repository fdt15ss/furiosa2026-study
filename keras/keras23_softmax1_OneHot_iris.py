import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_iris()
print(datasets)

print(datasets.DESCR)
x = datasets.data
y = datasets['target']

print(datasets.feature_names)
print(x.shape, y.shape) # (150, 4) (150,)

print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))
# one hot encoding 해야함

"""
[0,0,1,1,2] # (5,)
->
[[1,0,0]
[1,0,0]
[0,1,0]
[0,1,0]
[0,0,1]]    # (5,3)
"""
################# 원핫1. to_categorical #####################
# from keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) # (150, 3)

################# 원핫2. pandas #############################
# import pandas as pd

# y = pd.get_dummies(y, dtype=int)
# print(y)
# print(y.shape)


################# 원핫3. sklearn ############################
from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder() # 혼동행렬(sparse 형태)로 나온다.
encoder = OneHotEncoder(sparse_output=False)
# print(type(y)) # <class 'numpy.ndarray'>
# print(y.shape) # (150,)
y = y.reshape(-1, 1) # y.reshape(150, 1) 도 가능
# print(type(y)) # <class 'numpy.ndarray'>
# print(y.shape) # (150, 1)

# import os
# os.system("pause")

y = encoder.fit_transform(y)
# print(y)
# print(type(y)) # OneHotEncoder(sparse_output=True) 기본값이면 <class 'scipy.sparse._csr.csr_matrix'>
# print(y.shape) # (150, 3)
# y = y.toarray() OneHotEncoder에서 sparse_output=False였을 경우 toarray()로 변경해줘야 함
# print(type(y)) # <class 'numpy.ndarray'>
# print(y.shape) # (150, 3)

# exit()
# 통상적으로 원핫인코딩에서는 to_categorical로 바꿔주고 그다음 train_test_split을 사용하여 데이터를 나눠준다.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=333,
    shuffle=True,
    stratify=y,
)
print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)  # (120, 3) (30, 3)
# exit()

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss ='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(
    monitor = 'val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit(
    x_train,y_train, epochs=1000, batch_size=8,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1], 2))


y_predict = model.predict(x_test)

print('y_predict :', y_predict) # 수치형 자료
y_predict = np.argmax(y_predict, axis=1)
print('y_predict :', y_predict) # 문자형 자료
y_test = np.argmax(y_test, axis=1)
print('y_test :', y_test) # 문자형 자료

# exit()

# from keras.metrics import categorical_accuracy
# acc_score = np.mean(categorical_accuracy(y_test, y_predict)) # acc구하는 다른 방법
acc_score = accuracy_score(y_test, y_predict)
print('acc_score :', acc_score)
print("걸린시간 :", round(end_time-start_time, 2), '초')