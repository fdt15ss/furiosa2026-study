import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8],
                     ]).T
# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

imputer = SimpleImputer(strategy="most_frequent")
data2 = imputer.fit_transform(data)
print(data2)

imputer2 = SimpleImputer(strategy='constant', fill_value=777)
data3 = imputer2.fit_transform(data)
print(data3)


imputer3 = SimpleImputer(
    strategy=lambda x: np.min(x)
)
data4 = imputer3.fit_transform(data)
print(data4)