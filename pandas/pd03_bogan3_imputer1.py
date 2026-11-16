import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan],
                     ]).T
# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

SimpleImputer
imputer = SimpleImputer() # 디폴트 평균값.
data2 = imputer.fit_transform(data)
print(data2)

imputer2 = SimpleImputer(strategy='mean')
data3 = imputer2.fit_transform(data)
print(data3)

imputer3 = SimpleImputer(strategy='median')
data4 = imputer3.fit_transform(data)
print(data4)
