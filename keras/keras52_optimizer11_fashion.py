import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
import my_util
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime

# import matplotlib.pyplot as plt

#1. 데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(np.unique(y_train, return_counts=True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# exit()

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

x_augmented = datagen.flow(
    x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1),
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# 데이터 변환 완료. 데이터 합병
x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(-1,x_train.shape[1],x_train.shape[2],1)

x_train = np.concatenate((x_train/255.,x_augmented))
y_train = np.concatenate((y_train,y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
x_test = x_test/255.

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([10031,  9969,  9936, 10042, 10136,  9912, 10036, 10020,  9870,
#    10048], dtype=int64))

y_train = pd.get_dummies(y_train, dtype=int)
y_test = pd.get_dummies(y_test, dtype=int)


# [실습] 맹그러봐!!!
# 시작!!!! 기존 스코어보다 높일 것!!!
#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=x_test[0].shape, activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
from keras.optimizers import Adam
# learning_rate = 0.01
learning_rate = 0.001     # 디폴트
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=learning_rate), metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

save_path = './_save/mnist_fashion/' + datetime.now().strftime('%Y%m%d_%H%M%S')+ '_{epoch:04d}-{val_loss:.4f}.keras'
mcp = ModelCheckpoint(
    filepath = save_path,
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,

)

BATCH_SIZE = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=1, epochs=2000,
                    batch_size=BATCH_SIZE, validation_split=0.2,
                    # callbacks = [es]
                    callbacks = [es, mcp]

                    )

train_time = time.time() - start_time


#4. 평가 예측
loss, acc_eval = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = BATCH_SIZE,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = acc_eval,
    train_ration = 0,
    csv_file_path="11_fasion.csv"
)
