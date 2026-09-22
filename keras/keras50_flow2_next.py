import numpy as np

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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

augment_size = 100
# print(x_train[0][14])
# aaa = np.tile(x_train[0], augment_size)
# aaa = aaa.reshape(-1,28,28)
# print(aaa.shape)
# print(aaa[0][14])
# import matplotlib.pyplot as plt
# plt.imshow(aaa[4])
# plt.show()

xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False,
).next()

# print(xy_data)
# print(len(xy_data[0]))
# print(xy_data[0][0])
# print(type(xy_data))

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(xy_data[0][i],cmap="gray")
plt.show()