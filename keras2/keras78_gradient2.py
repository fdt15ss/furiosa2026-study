import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 6, 100)
print(x, len(x))

f = lambda x : x**2 - 4*x + 6

gradient = lambda x : 2*x -4

x = -100.0 # 초기값
epochs = 50
learning_rate = 0.1

print('epoch \t x \t f(x)')
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0, x, f(x)))

for i in range(epochs):
    x = x - learning_rate * gradient(x)

    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))