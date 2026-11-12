# 78_06 카피

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# SELU 고정 상수 (논문 제안값)
alpha = 1.6732632423543772
lmbda = 1.0507009873554805


# def selu(x, alpha, lmbda):
#     return lmbda * ((x>0) * x + (x<=0) * (alpha * (np.exp(x) - 1)))

selu = lambda x, alpha, lmbda : lmbda * ((x>0) * x + (x<=0) * (alpha * (np.exp(x) - 1)))

y = selu(x, alpha, lmbda)
plt.plot(x, y)
plt.grid()
plt.show()