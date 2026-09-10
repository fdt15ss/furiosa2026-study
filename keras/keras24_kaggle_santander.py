# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# path = 'C:/study/_data/kaggle_santander'
path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print(train_csv.shape)      #(200000, 201)
print(test_csv.shape)       #(200000, 200)
print(submission_csv.shape) #(200000, 1)

print(train_csv.info())
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([179902,  20098]))

# 판다스를 넘파이로 바꾸기
#1. y = y.to_numpy()
#2. y = np.array(y)

y = pd.get_dummies(y, dtype=int)

print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=337,
                                                    stratify=y)


#2. 모델구성
model = Sequential()
model.add(Dense(512, input_dim=200, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()
# exit()
#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


PATIENCE = 10
BATCH_SIZE = 256
es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=PATIENCE,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=500,
          batch_size=BATCH_SIZE,
          verbose=1,
          callbacks=[es],
          validation_split=0.2)
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred_example = model.predict(x_test)
print(y_pred_example[:10])

y_pred_example = np.round(y_pred_example)
print(y_pred_example[:10])
y_pred_example = np.argmax(y_pred_example, axis=1)
print(y_pred_example[:10])
y_test = np.argmax(y_test, axis=1)
print(y_test[:10])



acc_score = accuracy_score(y_test, y_pred_example)
print('acc_score :', acc_score)
# acc_score : 0.909425

y_pred = model.predict(test_csv)
y_pred = np.argmax(y_pred, axis=1)

submission_csv['target'] = y_pred

from datetime import datetime
date = datetime.now()
date = date.strftime("%Y%m%d_%H%M")

submission_csv.to_csv(path + f'submit/submission_{date}_p{PATIENCE}_b{BATCH_SIZE}_acc{acc_score:.4f}.csv')