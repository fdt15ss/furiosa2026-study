import numpy as np

x1 = np.array([1,2,3]) # (3, )
print("x1 = ", x1.shape)

x2 = np.array([[1,2,3]]) # (1, 3)
print("x2 = ", x2.shape)

x3 = np.array([[1,2],[3,4]]) # (2, 2)
print("x3 = ", x3.shape)

x4 = np.array([[1,2],[3,4],[5,6]]) # (3, 2)
print("x4 = ", x4.shape)

x5 = np.array([[[1,2],[3,4],[5,6]]]) # (1, 3, 2)
print("x5 = ", x5.shape)

x6 = np.array([[[1,2,],[3,4]],[[5,6,],[7,8]]]) # (2, 2, 2)
print("x6 = ", x6.shape)

x7 = np.array([[[[[1,2,3,4,5],[6,7,8,9,10]]]]]) # (1, 1, 1, 2, 5)
print("x7 = ", x7.shape)

x8 = np.array([[[1,2,3]],[[4,5,6]]]) # (2, 1, 3)
print("x8 = ", x8.shape)

x9 = np.array([[[[1]]],[[[2]]]]) # (2, 1, 1, 1)
print("x9 = ", x9.shape)

#데이터를 받을 때 shape를 잘 확인해야 함