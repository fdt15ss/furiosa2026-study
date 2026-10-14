import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG19(include_top=False, input_shape=(32,32,3)),
              ResNet50(include_top=False, input_shape=(32,32,3)),
              ResNet101(include_top=False, input_shape=(32,32,3)),
              DenseNet201(include_top=False, input_shape=(32,32,3)),
            #   InceptionV3(),
            #   InceptionResNetV2(),
              MobileNetV2(include_top=False, input_shape=(32,32,3)), 
              EfficientNetB0(include_top=False, input_shape=(32,32,3))]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random
import my_util

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.datasets import cifar10, cifar100

for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #스케일링
    x_train = x_train/255.
    x_test = x_test/255.

    #3. 컴파일 훈련
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    # 얼리스타핑 느세요
    es = EarlyStopping(monitor='val_loss',
        mode='min',
        patience=20,
        restore_best_weights=True) # 속 느세요
    # 리듀스엘알 느세요
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor= 0.5,
        patience= 10,
    )
    BATCH_SIZE = 32
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=100,
              batch_size= BATCH_SIZE,
              verbose=0,
              validation_split=0.2,
              callbacks=[es, rlr],
              )
    train_time = round(time.time() - start_time, 3)

    #4. 평가, 예측
    print("=======================================")
    print('걸린시간 : ', train_time, '초')
    loss = model.evaluate(x_test, y_test, verbose =0, batch_size=BATCH_SIZE//4)
    print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', loss[1])

    my_util.record_model_csv(
        model=model,
        data_shape=x_train.shape,
        random_num=333,
        batch_size=BATCH_SIZE,
        history=history,
        training_time=train_time,
        test_loss= loss[0],
        sub_score= loss[1],
        train_ration=0,
        base_model=model_i.name,
        csv_file_path="13_cifar10.csv"
    )

    