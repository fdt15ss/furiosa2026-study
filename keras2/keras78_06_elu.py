# 78_05 카피

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def elu(x, alpha):
#     return (x>0)*x + (x<0)*(alpha*(np.exp(x) - 1))

alpha = 1
elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x) - 1))

y = elu(x, alpha)
plt.plot(x, y)
plt.grid()
plt.show()