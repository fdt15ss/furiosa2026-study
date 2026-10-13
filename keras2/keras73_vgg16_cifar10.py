import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import my_util
import time
from keras.callbacks import EarlyStopping

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# 데이터 증폭
datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    rescale=1./255,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,     # 상하 반전
    width_shift_range= 0.1, # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range= 20,      # 각도 조절
    zoom_range=0.1,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest",
)

augment_size = 40000

# xy_data = datagen.flow(
#     np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),
#     np.zeros(augment_size),
#     batch_size=augment_size,
#     shuffle=False,
# ).next()

randidx = np.random.randint(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = next(
    datagen.flow(
        x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3),
        y_augmented,
        batch_size=augment_size,
        shuffle=False,
    )
)[0]

# 데이터 변환 완료. 데이터 합병
x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(-1,x_train.shape[1],x_train.shape[2],3)

x_train = np.concatenate((x_train/255.,x_augmented))
y_train = np.concatenate((y_train,y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
x_test = x_test/255.

print(np.unique(y_train, return_counts=True))


ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))


#2. 모델
# model = VGG16()
####### 디폴트 ##########
vgg16= VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)
# vgg16.trainable = False
vgg16.trainable = True
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

# 실습 #
# 비교할 거
#1. 이전에 본인이 한 최상의 결과.
#2. 가중치를 동결하고 훈련시켰을때, trainable = False
#3. 가중치를 동결하지 않고 훈련시켰을때, trainable = True
#4. 시간까지 비교.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
BATCH_SIZE = 256

es = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
)


start_time = time.time()
history = model.fit(
    x_train, y_train, batch_size= BATCH_SIZE, epochs=2000,
    validation_split=0.2,
    callbacks=[es]
)
train_time = round(time.time() - start_time, 3)

results = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", results[0])
print("acc : ", results[1])

print("걸린 시간 : ",train_time , "초")

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 42,
    batch_size = BATCH_SIZE,
    history = history,
    training_time = train_time,
    test_loss = results[0],
    sub_score = results[1],
    train_ration = 0,
    csv_file_path="13_cifar10.csv"
)