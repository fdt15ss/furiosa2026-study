# 11-4 카피
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)


#1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
model = DecisionTreeClassifier()

#3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['acc'])
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)

result = model.score(x_data, y_data)
print("acc : ", result)


acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_score :", acc)