# acc : 0.93
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

#1. 데이터
datasets = fetch_covtype()
# print(datasets)
x = datasets.data # (581012, 54)
y = datasets.target

print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
y_cat = to_categorical(y) # 
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.8,
                                                    stratify=y,
                                                    random_state = 333)

print('x_train.shape :', x_train.shape)     # x_train.shape : (464809, 54)
print('x_test.shape :', x_test.shape)       # x_test.shape : (116203, 54)
print('y_train.shape :', y_train.shape)     # y_train.shape : (464809, 7)
print('y_test.shape :', y_test.shape)       # y_test.shape : (116203, 7)
print('y_cat.shape :', y_cat.shape)         # y_cat.shape : (581012, 8)
print(y_cat[:10])
print(y_test[:10])
exit()
#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=54, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train,
          epochs = 2000,
          callbacks=[es],
          validation_split=0.2,
          batch_size=4096
          )

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print(f'loss : {round(result[0], 4)}, acc: {round(result[1], 4)}')

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict)
# y_test = np.argmax(y_test)
# acc = accuracy_score(y_test, y_predict)

acc = np.mean(categorical_accuracy(y_test, y_predict))
print('acc :', acc)

# loss : 0.1365, acc: 0.9502
# acc : 0.95019925