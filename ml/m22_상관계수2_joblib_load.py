import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import joblib, pickle

#1 데이터
path = './_save/m22/'
df = joblib.load(path + 'm22_iris_df.joblib')
print(df)

print("====================== 상관관계 히트맵 =====================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# square : 정사각형으로 표시
# annot : 숫자 표시
# cbar : 컬러바 표시

plt.show()