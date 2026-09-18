# https://drive.google.com/drive/folders/1XnXBR2rdrHf9GLEgMyDFq9Ellp5DeX2P

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import time

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,   # 수평 뒤집기,
    # vertical_flip=True,     # 수직 뒤집기 (상하 반전)
    # width_shift_range=0.1,  # 평행이동
    # height_shift_range=0.1,
    # rotation_range=5,   # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7,    # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # fill_mode = 'nearest'
)
test_datagen = ImageDataGenerator( # 시험지는 그대로 둬야함. 테스트 데이터는 변환하지 않음. 
    rescale = 1./255,
)

path_train = "./_data/image/brain/train/"
path_test = "./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
   path_train, # 경로
   target_size=(150, 150),
   batch_size=160,
   class_mode='binary', # 이진분류
   color_mode='grayscale', #흑백
   shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size = (150, 150),
    batch_size=120,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False, # test에서는 필요가 없다.
)
# Found 120 images belonging to 2 classes.
print(xy_train[0][0].shape) # (160, 150, 150, 1)
print(xy_train[0][1].shape) # (160,)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)

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
model.add(Conv2D(128,(2,2), activation='relu'))
model.add(Conv2D(256,(2,2), activation='relu'))
model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())
model.add(Flatten())

model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
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
start_time = time.time()

model.fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 1500,
    validation_split=0.2,
    callbacks=[es],

)
end_time = time.time()

#4. 평가, 예측

print('걸린시간 : ', round(end_time - start_time, 3), '초')
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('accuracy : ', result[1])


# Epoch 253/500
# (16-16-1)걸린시간 :  47.64 초
# (16-16-1)loss :  0.387095183134079
# (16-16-1)accuracy :  0.8500000238418579

# Epoch 137/500
# (8-8-1)걸린시간 :  27.613 초
# (8-8-1)loss :  0.569293200969696
# (8-8-1)accuracy :  0.6916666626930237

# Epoch 85/1500
# 걸린시간 :  18.706 초
# loss :  0.6093124151229858
# accuracy :  0.675000011920929

# Epoch 98/1500
# 걸린시간 :  20.65 초
# loss :  0.6104902625083923
# accuracy :  0.675000011920929

# Epoch 96/1500
# (Flatten)걸린시간 :  20.648 초
# (Flatten)loss :  0.01788564771413803
# (Flatten)accuracy :  0.9916666746139526

# Epoch 49/1500
# 걸린시간 :  15.464 초
# loss :  0.008990599773824215
# accuracy :  1.0