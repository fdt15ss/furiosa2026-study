import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    print('1사분위 :', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print("IQR :", iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data > upper_bound) | (data < lower_bound)), \
    iqr, lower_bound, upper_bound

outlier_loc1, iqr1, lower_bound1, upper_bound1 = outlier(aaa.T[0])
outlier_loc2, iqr2, lower_bound2, upper_bound2 = outlier(aaa.T[1])
print('이상치의 위치1 :', outlier_loc1)
print('이상치의 위치2 :', outlier_loc2)
# import pandas as pd

# series_aaa1 = pd.Series(aaa.T[0])
# series_aaa2 = pd.Series(aaa.T[1])
# series_aaa1[outlier_loc1] = np.nan
# series_aaa2[outlier_loc2] = np.nan
aaa_clipped1 = np.clip(aaa.T[0], lower_bound1, upper_bound1)
aaa_clipped2 = np.clip(aaa.T[1], lower_bound2, upper_bound2)

print(aaa_clipped1)
print(aaa_clipped2)

# aaa.T[0][outlier_loc1] = np.nan # 에러
# print(series_aaa1)
# print(series_aaa2)

import matplotlib.pyplot as plt

# 1행 2열로 화면 분할 (상하로 배치하려면 plt.subplots(2, 1)로 변경)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 첫 번째 그래프 (왼쪽)
axes[0].boxplot(aaa_clipped1)
axes[0].axhline(upper_bound1, color='pink', label='upper bound')
axes[0].axhline(lower_bound1, color='pink', label='lower bound')
axes[0].set_title('Data 1')
axes[0].legend()

# 두 번째 그래프 (오른쪽)
axes[1].boxplot(aaa_clipped2)
axes[1].axhline(upper_bound2, color='pink', label='upper bound')
axes[1].axhline(lower_bound2, color='pink', label='lower bound')
axes[1].set_title('Data 2')
axes[1].legend()

plt.tight_layout()  # 그래프 간격 자동 조절
plt.show()