# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from datetime import datetime
import my_util
from sklearn.model_selection import train_test_split
import pandas as pd

npy_path = '_data\\men_women_npy\\'
x_train = np.load(npy_path + 'keras46_men_women_x_train.npy') 
y_train = np.load(npy_path + 'keras46_men_women_y_train.npy') 
x_test = np.load(npy_path + 'keras46_men_women_x_test.npy') 
y_test = np.load(npy_path + 'keras46_men_women_y_test.npy') 

print(pd.DataFrame(y_train).value_counts())
# exit()

#2. 모델구성
# 실습 : "맹그러봐!!"
# acc 1.0

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(x_train[0].shape), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64,(2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(2,2), activation='relu'))
model.add(GlobalAveragePooling2D())
# model.add(Flatten())

# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
# exit()
#3 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    patience = 20,
    mode = 'auto',
    restore_best_weights=True,
)

save_path = './_save/men_women/' + datetime.now().strftime('%Y%m%d_%H%M%S')+ '_{epoch:04d}-{val_loss:.4f}.keras'
mcp = ModelCheckpoint(
    filepath = save_path,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,

)
BATCH_SIZE = 16

start_time = time.time()
history = model.fit(
    x_train, y_train,
    batch_size = BATCH_SIZE, # 1024 oom, 128 oom, 64 느림, 32 느려짐,
    epochs = 1000,
    validation_split=0.2,
    callbacks=[es, mcp],

)
end_time = time.time()

#4. 평가, 예측
train_time = round(end_time - start_time, 3)
print('걸린시간 : ', train_time, '초')
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('accuracy : ', result[1])

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = BATCH_SIZE,
    history = history,
    training_time = train_time,
    test_loss = result[0],
    sub_score = result[1],
    train_ration = 0,
    csv_file_path="men_women.csv"
)

# my_util.leaveTop(path=save_path, prefix=datetime.now().strftime('%Y%m%d_%H%M%S'), subfix=".keras", count=5, mode="min")

# Epoch 85/1500
# 걸린시간 :  1408.079 초
# loss :  0.4175207316875458
# accuracy :  0.8344043493270874