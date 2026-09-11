# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=337,
                                                    stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

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
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


PATIENCE = 20
BATCH_SIZE = 512
es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=PATIENCE,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=BATCH_SIZE,
          verbose=1,
          callbacks=[es],
          validation_split=0.2)
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss, 'acc :', acc)

y_pred_example = model.predict(x_test)
print(y_pred_example[:10])

y_pred_example = np.round(y_pred_example)
print(y_pred_example[:10])

acc_score = accuracy_score(y_test, y_pred_example)
print('acc_score :', acc_score)

y_pred = model.predict(test_csv)
# y_pred = np.round(y_pred)

submission_csv['target'] = y_pred

from datetime import datetime
date = datetime.now()
date = date.strftime("%Y%m%d_%H%M")

submission_csv.to_csv(path + f'submit/submission_{date}_p{PATIENCE}_b{BATCH_SIZE}_acc{acc_score:.4f}.csv')

# acc_score(softmax2) : 0.909425
# acc_score(sigmoid) : 0.911525
# acc_score(MinMaxScaler,  acc: 0.8992 - loss: 0.3268 - val_acc: 0.9007 - val_loss: 0.3236 문제 발생) : 0.8995
# acc_score : 0.914475
# acc_score : 0.91405
# acc_score(반올림 안함, restore_best_weights=False) : 0.913
# acc_score(StandardScaler 적용) : 0.911625
# acc_score(restore_best_weights=False) : 0.881575
# acc_score(MaxAbsScaler 적용) : 0.907775
# acc_score : 0.913525
# acc_score(RobustScaler 적용) : 0.911875
########################### Robust 여기까지함