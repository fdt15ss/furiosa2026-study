from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = "c:/study/_data/image/"

img = load_img(path + 'pic.jpg', target_size=(200, 200))

print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x2C809B6FA90>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr) 
print(arr.shape) # (150, 150, 3)
print(type(arr)) # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) # 차원 증가
print(arr)
print(arr.shape) # (1, 150, 150, 3)


np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras48_me.npy', arr=arr)
# np.save(np_path + 'keras48_dog.npy', arr=arr)
# np.save(np_path + 'keras48_cat.npy', arr=arr)