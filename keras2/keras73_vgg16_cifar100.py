import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import my_util
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#################### 데이터 증폭 ####################
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000

randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3
)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# 스케일링
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# sparse_categorical_crossentropy는 원-핫 인코딩 없이 정수 라벨(Integer)을 그대로 사용합니다.
print(y_train.shape, y_test.shape)  # (90000, 1) (10000, 1)


# 2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)
# vgg16.trainable = False
vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. 컴파일 및 훈련 설정
# sparse_categorical_crossentropy 적용
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

path = './_save/14_cifar100/'
date = datetime.datetime.now().strftime("%y%m%d_%H%M")
filename = '.keras'
filepath = "".join([path, "k73_14_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)

BATCH_SIZE = 1024
start_time = time.time()
history = model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp]
)
train_time = round(time.time() - start_time, 3)

# 4. 평가
results = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", results[0])
print("acc : ", results[1])
print("걸린 시간 : ", train_time, "초")

my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=42,
    batch_size=BATCH_SIZE,
    history=history,
    training_time=train_time,
    test_loss=results[0],
    sub_score=results[1],
    train_ration=0,
    csv_file_path="14_cifar100.csv"
)