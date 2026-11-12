
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)

# 활성화 함수 정의 (Swish는 beta=1일 때 SiLU와 동일)
mish = lambda x: x * np.tanh(np.log1p(np.exp(x)))  # log(1 + exp(x))는 log1p가 수치적으로 더 안정적
swish = lambda x, beta=1.0: x * (1 / (1 + np.exp(-beta * x)))

y_mish = mish(x)
y_swish = swish(x, beta=1.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# 1. 함수 값 비교
ax1.plot(x, y_mish, label='Mish', linewidth=2)
ax1.plot(x, y_swish, label='Swish (SiLU, β=1)', linestyle='--', linewidth=2)
ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.axvline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.set_title('Mish vs Swish Activation Functions')
ax1.set_ylabel('Output')
ax1.legend()
ax1.grid(True, alpha=0.4)

# 2. 두 함수의 차이 (Mish - Swish)
ax2.plot(x, y_mish - y_swish, color='purple', label='Mish - Swish')
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_title('Difference (Mish - Swish)')
ax2.set_xlabel('x')
ax2.set_ylabel('Difference')
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()