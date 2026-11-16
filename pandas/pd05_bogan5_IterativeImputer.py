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

from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=333,
)

imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,
    # tol=0.001, # 디폴트 (얼리 스탑의 기준)
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2)

print("실제 반복 횟수:", imputer.n_iter_)