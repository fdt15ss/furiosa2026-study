from sklearn.datasets import load_iris
import pandas as pd
import joblib, pickle

#1 데이터
datasets = load_iris()

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
df['target'] = y
print(df)

# 맹그러봐!!!
# joblib, 핑클로 세이브하기!!!
path = './_save/m22/'
joblib.dump(df, path + 'm22_iris_df.joblib')
pickle.dump(df, open(path + 'm22_iris_df.pickle', 'wb'))
