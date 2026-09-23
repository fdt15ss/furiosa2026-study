# 실습
# 여자 데이터 증폭해서, 성능 올려봐!!!!

# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from datetime import datetime
import my_util
from sklearn.model_selection import train_test_split
import pandas as pd

# val_acc이 너무 낮은 경우 리스케일 되어 저장되어있는 것을 한번더 리스케일하지 않았나 고민해볼 것.
npy_path = '_data\\men_women_npy\\'
x_train = np.load(npy_path + 'keras46_men_women_x_train.npy') 
y_train = np.load(npy_path + 'keras46_men_women_y_train.npy') 
x_test = np.load(npy_path + 'keras46_men_women_x_test.npy') 
y_test = np.load(npy_path + 'keras46_men_women_y_test.npy') 

# 데이터 증폭
datagen = ImageDataGenerator(
    # rescale=1/255.,
    #증폭 변환하는 파라미터들
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,     # 상하 반전
    width_shift_range= 0.1, # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range= 20,      # 각도 조절
    zoom_range=0.1,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest",
)

counts = pd.DataFrame(y_train).value_counts()
print(abs(counts[1]-counts[0]))
augment_size = abs(counts[1]-counts[0])

x_train_women = x_train[np.where(y_train >0.0)]
y_train_women = y_train[np.where(y_train >0.0)]

randidx = np.random.randint(x_train_women.shape[0],size=augment_size)

x_augmented = x_train_women[randidx].copy()
y_augmented = y_train_women[randidx].copy()
x_augmented = datagen.flow(
    x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],3),
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]


x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(-1,x_train.shape[1],x_train.shape[2],3)

x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))
print(np.max(y_train))
# exit()
print(x_train.shape, y_train.shape) # (10364, 200, 200, 3) (10364,)
# x_test = x_test/255.
print(pd.DataFrame(y_train).value_counts())


#2. 모델구성
# 실습 : "맹그러봐!!"
# acc 1.0

# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape=(x_train[0].shape), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# model.add(Conv2D(64,(2,2), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))
# model.add(Conv2D(128,(2,2), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))
# model.add(Conv2D(128,(2,2), activation='relu'))
# model.add(GlobalAveragePooling2D())
# # model.add(Flatten())

# # model.add(Dense(64, activation='relu'))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# # model.add(Dense(16, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()
# # exit()
# #3 컴파일, 훈련
# from keras.optimizers import Adam
# # learning_rate = 0.01
# # learning_rate = 0.001     # 디폴트
# learning_rate = 0.0005
# # learning_rate = 0.005
# # learning_rate = 0.05
# # learning_rate = 0.009

# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=learning_rate),
#               metrics=['acc'])

# es = EarlyStopping(
#     monitor = 'val_loss',
#     patience = 20,
#     mode = 'auto',
#     restore_best_weights=True,
# )

# save_path = './_save/men_women/' + datetime.now().strftime('%Y%m%d_%H%M%S')+ '_{epoch:04d}-{val_loss:.4f}.keras'
# mcp = ModelCheckpoint(
#     filepath = save_path,
#     monitor = 'val_loss',
#     mode = 'auto',
#     save_best_only=True,

# )
# BATCH_SIZE = 4

# start_time = time.time()
# history = model.fit(
#     x_train, y_train,
#     batch_size = BATCH_SIZE, # 1024 oom, 128 oom, 64 느림, 32 느려짐,
#     epochs = 1000,
#     validation_split=0.2,
#     callbacks=[es, mcp],

# )
# end_time = time.time()
# train_time = round(end_time - start_time, 3)
model = load_model('_save\\men_women\\20260923_140316_0072-0.0445.keras')

#4. 평가, 예측
# print('걸린시간 : ', train_time, '초')
result = model.evaluate(x_test, y_test, batch_size=8)
print('loss : ', result[0])
print('accuracy : ', result[1])

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    # batch_size = BATCH_SIZE,
    batch_size = 4,
    history = None,
    # history = history,
    # training_time = train_time,
    training_time = 2523.505,
    test_loss = result[0],
    sub_score = result[1],
    train_ration = 0,
    csv_file_path="15_men_women.csv"
)

# my_util.leaveTop(path=save_path, prefix=datetime.now().strftime('%Y%m%d_%H%M%S'), subfix=".keras", count=5, mode="min")

# Epoch 85/1500
# 걸린시간 :  1408.079 초
# loss :  0.4175207316875458
# accuracy :  0.8344043493270874

# 걸린시간 : 2712.563초
# loss :  0.30996227264404297
# accuracy :  0.8849999904632568