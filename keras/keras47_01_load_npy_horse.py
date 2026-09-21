# https://drive.google.com/drive/folders/1XnXBR2rdrHf9GLEgMyDFq9Ellp5DeX2P

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from datetime import datetime
import my_util
import pandas as pd

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     # horizontal_flip=True,   # 수평 뒤집기,
#     # vertical_flip=True,     # 수직 뒤집기 (상하 반전)
#     # width_shift_range=0.1,  # 평행이동
#     # height_shift_range=0.1,
#     # rotation_range=5,   # 각도조절(정해진 각도만큼 이미지 회전)
#     # zoom_range=1.2,
#     # shear_range=0.7,    # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
#     # fill_mode = 'nearest'
# )

# test_datagen = ImageDataGenerator( # 시험지는 그대로 둬야함. 테스트 데이터는 변환하지 않음. 
#     rescale = 1./255,
# )

# path_train = "./_data/image/cat_dog/training_set/"
# path_test = "./_data/image/cat_dog/test_set/"

# xy_train = train_datagen.flow_from_directory(
#    path_train, # 경로
#    target_size=(200, 200),
#    batch_size=10000,
#    class_mode='binary', # 이진분류
#    color_mode='rgb', 
#    shuffle=True,
# )
# # Found 160 images belonging to 2 classes.

# xy_test = test_datagen.flow_from_directory(
#     path_test,
#     target_size = (200, 200),
#     batch_size=10000,
#     class_mode='binary', # 이진분류
#     color_mode='rgb', #흑백
#     shuffle=False, # test에서는 필요가 없다.
# )
# # Found 120 images belonging to 2 classes.
# print(xy_train[0][0].shape) # (160, 150, 150, 1)
# print(xy_train[0][1].shape) # (160,)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]
npy_path = './_data/horse-human_npy/'

x_train = np.load(npy_path + 'keras46_horse-human_x_train.npy')
y_train = np.load(npy_path + 'keras46_horse-human_y_train.npy')
x_test = np.load(npy_path + 'keras46_horse-human_x_test.npy')
y_test = np.load(npy_path + 'keras46_horse-human_y_test.npy')

print(x_train.shape, y_train.shape) # 
print(x_test.shape, y_test.shape)   # 



# exit()
#2. 모델구성
# 실습 : "맹그러봐!!"
# acc 1.0

model = Sequential()
model.add(Conv2D(32, (4,4), input_shape=(x_train[0].shape), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(4,4))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
# model.add(Flatten())

# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()
# exit()
#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    patience = 20,
    mode = 'auto',
    restore_best_weights=True,
)

save_path = './_save/horse-human/' + datetime.now().strftime('%Y%m%d_%H%M%S')+ '_{epoch:04d}-{val_loss:.4f}.keras'
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
    epochs = 1500,
    validation_split=0.2,
    callbacks=[es, mcp],
    # callbacks=[es, ],
    

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
    csv_file_path="horse-human.csv"
)


# Epoch 85/1500
# 걸린시간 :  1408.079 초
# loss :  0.4175207316875458
# accuracy :  0.8344043493270874