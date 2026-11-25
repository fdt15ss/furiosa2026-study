# ML 8일차 정리 — 모델 저장(joblib · pickle · XGBoost) · `**kwargs` · 상관계수

> 녹취 기반 정리. 오늘 핵심은
> 1. XGBoost 하이퍼파라미터를 **딕셔너리 + `**` 언패킹**으로 넣기, Early Stopping
> 2. 학습한 모델(가중치)을 **저장하고 다시 불러오기** — joblib / pickle / XGBoost 자체 저장
> 3. pandas DataFrame 저장, **상관계수(`df.corr()`) + seaborn 히트맵**으로 피처 판단하기

---

## 0. 오늘의 흐름 한눈에 보기

| 순서 | 주제 | 파일 |
|---|---|---|
| 1 | XGBoost 파라미터 정리 + `**params` + Early Stopping + joblib 저장 | `m19_joblib_01_save.py` |
| 2 | joblib 불러오기 | `m19_joblib_02_load.py` |
| 3 | pickle 저장 / 불러오기 | `m20_pickle_01_save.py`, `m20_pickle_02_load.py` |
| 4 | XGBoost 자체 `save_model` / `load_model` | `m21_xgb_01_save.py`, `m21_xgb_02_load.py` |
| 5 | DataFrame 만들기 → joblib 저장/로드 → 상관계수 히트맵 | `m22_...` |

---

## 1. 왜 정형 데이터에선 여전히 ML 3대장인가

- 데이터 분석(정형 데이터)만 보면 **XGBoost·LightGBM·CatBoost 같은 트리/부스팅 계열이 훨씬 강력**하다.
- TensorFlow 같은 딥러닝으로는 정형 데이터 분석이 오히려 어렵다.
- 시계열은 "LSTM·트랜스포머가 최고"라고 알려져 있지만, **증권·은행·파생상품·발전소** 쪽은 지금도 3대장 위주로 쓴다.
- → 회사에서 데이터 분석을 맡게 되면 ML을 확실히 알고 있어야 한다.

---

## 2. XGBoost 하이퍼파라미터

### 2-1. 자주 쓰는 파라미터
```python
params = {
    'n_estimators': 1000,       # 트리(부스팅 라운드) 개수 = 몇 번 훈련할지
    'learning_rate': 0.1,       # 학습률 — 가장 중요
    'max_depth': 6,             # 트리 깊이 — 가장 중요
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'reg_alpha': 0,             # L1 규제
    'reg_lambda': 1,            # L2 규제
    'random_state': 3377,       # 이것도 의외로 중요
}
```
- **`n_estimators`, `learning_rate`, `max_depth` 세 개는 무조건 잘 먹힌다.** 나머지는 조합하며 찾아보면 된다.
- `reg_alpha` / `reg_lambda`는 **L1 / L2 규제** 개념이라, 제대로 하려면 그것만으로 하루가 필요하다. 오늘은 "이런 게 있다" 정도만.
- 무엇이 무조건 좋다는 건 없다. 많이 써봐야 감이 생긴다.
- 튜닝을 자동으로 하고 싶으면 **Optuna** 같은 도구를 쓴다.

### 2-2. 딕셔너리를 모델에 넣기 → `**` 언패킹
```python
from xgboost import XGBRegressor

model = XGBRegressor(**params)   # 딕셔너리를 키워드 인자로 풀어서 전달
# model = XGBRegressor(n_estimators=1000, learning_rate=0.1, ...) 와 동일
```
- 파라미터를 하나하나 길게 쓰는 것보다 **딕셔너리로 빼두는 쪽이 가독성이 훨씬 좋다.**

### 2-3. `set_params`로 나중에 넣기 + Early Stopping
```python
model = XGBRegressor()
model.set_params(**params, early_stopping_rounds=10)

model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],   # Early Stopping 판단 기준 데이터
          verbose=1)
```
- Early Stopping은 **무엇을 기준으로 멈출지**가 필요하다 → `fit`의 `eval_set`으로 지정한다.
- `eval_set`은 **(X, y) 튜플의 리스트**로 넣는다. 여러 개를 넣을 수 있어서 리스트다.
- 수업 결과: `n_estimators=2000`으로 줬는데 **125번째에서 멈춤**.
- 캘리포니아 기준 XGB 디폴트 **약 0.83**. 앞에서 본 보팅·배깅·스태킹 최고점(약 0.84)을 기준점으로 잡고 튜닝을 해봤다.

> ⚠️ 보충: XGBoost 2.x부터 `early_stopping_rounds`는 `fit()`이 아니라 **생성자 또는 `set_params`** 에 넣는다. 수업 코드처럼 `set_params`에 넣는 방식이 맞다.

---

## 3. 파이썬 기초: `*args`와 `**kwargs` (별 하나, 별 두 개)

> "별 두 개는 **딕셔너리**를 땡겨온다, 별 하나는 **튜플**을 땡겨온다. 끝."

| 표기 | 받는 형태 | 용도 |
|---|---|---|
| `*args` | 튜플 | 개수가 정해지지 않은 **위치 인자** |
| `**kwargs` | 딕셔너리 | 개수가 정해지지 않은 **키워드 인자** (key=value) |

### 함수 정의할 때 (받기)
```python
def f(*args, **kwargs):
    print(args)     # (1, 2, 3)            ← 튜플
    print(kwargs)   # {'a': 10, 'b': 20}   ← 딕셔너리

f(1, 2, 3, a=10, b=20)
```

### 함수 호출할 때 (풀어서 넣기 = 언패킹)
```python
nums = (1, 2, 3)
opts = {'a': 10, 'b': 20}

f(*nums)     # f(1, 2, 3)
f(**opts)    # f(a=10, b=20)
XGBRegressor(**params)   # ← 오늘 쓴 바로 그 문법
```
- 이 문법을 보면 AI 석사까지 나온 사람들도 많이 굳는다. **외워두기.**
- 강사님 조언: 요즘은 AI가 코드를 잘 짜주지만 **코드를 이해하고 AI를 쓰는 사람**과 그냥 복붙하는 사람의 차이는 2~3년 뒤에 크게 벌어진다. 이해하지 못하면 고장 났을 때 유지보수를 못 한다.

---

## 4. 모델 저장 ① — joblib

- ML·데이터 분석 쪽 저장은 거의 **joblib 아니면 pickle**로 한다.
- 컴피티션에서 "가중치 파일을 내라"고 하면 대부분 **joblib 파일**이다.
- **모델(가중치)뿐 아니라 데이터도** 저장할 수 있다. 압축률이 좋고 불러오는 속도도 빠르다.
- joblib은 scikit-learn을 설치할 때 **같이 설치**되어 있다.

### 4-1. 저장 (`m19_joblib_01_save.py`)
```python
import joblib

model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)
print('R2:', model.score(x_test, y_test))

path = './_save/m19_joblib/'          # 상대 경로 (폴더는 미리 만들어 둠)
joblib.dump(model, path + 'm19_joblib1_save.joblib')
```
- 형식: `joblib.dump(저장할 객체, 경로+파일명)`
- 가중치는 **모델 안에** 들어 있으므로 `model`을 그대로 저장하면 된다.
- 확장자는 아무거나 써도 동작한다. `.joblib`처럼 쓰는 건 **가독성** 때문이다.

### 4-2. 불러오기 (`m19_joblib_02_load.py`)
```python
import joblib
from sklearn.metrics import r2_score

# 데이터 로드·분리·스케일링은 저장할 때와 동일하게
# 모델 정의 / 파라미터 / 훈련 코드는 필요 없음

path = './_save/m19_joblib/'
model = joblib.load(path + 'm19_joblib1_save.joblib')

y_pred = model.predict(x_test)
print('R2:', r2_score(y_test, y_pred))   # 저장 때와 같은 점수가 나와야 정상
```

> 상대 경로 vs 절대 경로
> - 절대 경로: `C:/...`, `D:/...`처럼 드라이브부터 쓰는 경로
> - 상대 경로: 현재 위치 기준 `./_save/...`

---

## 5. 모델 저장 ② — pickle

joblib과 거의 쌍벽이다. 요즘은 joblib이 조금 더 낫다고 하지만 방식은 거의 같다. 차이는 **파일을 `open()`으로 열어야 한다**는 것뿐이다.

### 5-1. 저장 (`m20_pickle_01_save.py`)
```python
import pickle

path = './_save/m20_pickle/'
pickle.dump(model, open(path + 'm20_pickle1_save.pickle', 'wb'))   # wb = write binary
```

### 5-2. 불러오기 (`m20_pickle_02_load.py`)
```python
import pickle

path = './_save/m20_pickle/'
model = pickle.load(open(path + 'm20_pickle1_save.pickle', 'rb'))  # rb = read binary
```

| | joblib | pickle |
|---|---|---|
| 저장 | `joblib.dump(model, path)` | `pickle.dump(model, open(path, 'wb'))` |
| 로드 | `joblib.load(path)` | `pickle.load(open(path, 'rb'))` |
| 특징 | 대용량·numpy 배열에 강함, 압축 | 파이썬 표준 라이브러리 |

> 보충: 파일을 닫는 것까지 깔끔하게 하려면 `with` 문을 쓰는 게 정석이다.
> ```python
> with open(path, 'wb') as f:
>     pickle.dump(model, f)
> with open(path, 'rb') as f:
>     model = pickle.load(f)
> ```

> 강사님 에피소드: Keras는 원래 `.h5`(HDF5)로 저장했는데, 버전이 바뀌면서 `.keras` 확장자로 바뀌었다. 라이브러리 버전이 자주 바뀌니 **안 되면 에러 메시지를 읽고 맞춰주면 된다.**

---

## 6. 모델 저장 ③ — XGBoost 자체 저장

XGBoost 자체에도 저장/로드 기능이 있다. 다만 다른 모델들과 같이 쓰는 경우가 많아서 **실무에선 joblib·pickle을 더 많이 쓴다.**

### 6-1. 저장 (`m21_xgb_01_save.py`)
```python
path = './_save/m21_xgb/'
model.save_model(path + 'm21_xgb1_save.json')
```
- 지원 포맷: **JSON** 또는 **UBJSON(`.ubj`, 기본값)**
- 다른 확장자(`.dat` 등)로 저장하면 "UBJSON 포맷으로 저장한다"는 경고가 뜬다. 수업에서는 경고를 꺼서 해결했다.

```python
import warnings
warnings.filterwarnings('ignore')
```
> 보충: 경고를 끄는 대신 확장자를 `.json`이나 `.ubj`로 쓰면 경고 자체가 안 난다.

### 6-2. 불러오기 (`m21_xgb_02_load.py`)
```python
from xgboost import XGBRegressor

model = XGBRegressor()                       # ① 빈 모델 먼저 정의 (joblib/pickle과 다른 점)
model.load_model(path + 'm21_xgb1_save.json')  # ② 가중치 불러오기

print('R2:', model.score(x_test, y_test))
```
- joblib/pickle은 `joblib.load()`가 모델 객체 자체를 돌려준다.
- XGBoost는 **모델을 먼저 만들고 → `.load_model()`** 해야 한다.

### 오늘 저장법 3종 요약
| 방법 | 저장 | 로드 | 사용 빈도 |
|---|---|---|---|
| joblib | `joblib.dump(model, path)` | `model = joblib.load(path)` | ★★★ (가장 많이) |
| pickle | `pickle.dump(model, open(path,'wb'))` | `model = pickle.load(open(path,'rb'))` | ★★ |
| XGBoost | `model.save_model(path)` | `model = XGBRegressor(); model.load_model(path)` | ★ |

---

## 7. sklearn 데이터셋 불러오기: 딕셔너리 vs 객체

```python
from sklearn.datasets import load_iris
datasets = load_iris()

x = datasets['data']     # 딕셔너리 키 방식
y = datasets['target']

x = datasets.data        # 클래스 객체 속성 방식
y = datasets.target

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```
- sklearn 데이터셋(`Bunch`)은 **두 방식 모두** 쓸 수 있게 만들어져 있다.
- 일반적으로는 `obj['key']`가 되면 **딕셔너리(데이터)**, `obj.attr`이 되면 **객체**라고 판단하면 된다.
- LLM(LangChain) 작업할 때도 이 구분을 엄청 많이 쓴다. 헷갈리면 둘 다 해보기.

---

## 8. pandas DataFrame 만들기 · 저장 · 컬럼 추가

### 8-1. DataFrame 생성
```python
import pandas as pd

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)          # 150 x 4
```
- `x`는 행렬 형태 데이터, `columns`는 리스트 → 순서대로 컬럼 이름이 붙는다.
- 인덱스는 지정하지 않으면 자동으로 0부터 생성된다.

### 8-2. 컬럼(타겟) 추가
```python
df['Target'] = y
print(df.shape)    # (150, 5)
```
- numpy 배열, Series, 리스트 모두 **`df['컬럼명'] = 값`** 으로 바로 추가된다.

### 8-3. DataFrame도 joblib으로 저장/로드
```python
import joblib
path = './_save/m22/'
joblib.dump(df, path + 'm22_iris_df.joblib')

df = joblib.load(path + 'm22_iris_df.joblib')
print(df.shape)    # (150, 5)
```
- **장점:** numpy(`.npy`)로 저장하면 인덱스와 컬럼명이 빠지지만, **joblib/pickle은 인덱스·컬럼명까지 통째로** 저장된다.
- 큰 CSV를 받아서 DataFrame으로 작업한 뒤 joblib으로 저장해 옮기면 깔끔하다.
- 컴피티션에서 데이터 파일을 joblib으로 제출하라는 경우도 있다.

---

## 9. 상관계수 (Correlation)

### 9-1. 왜 보나?
- 전에 배운 **Feature Importance**는 트리·부스팅 모델이 주는 중요도라서 **모델에 종속적**이고, 모델이 과적합되었을 수도 있다.
- 지표 하나만 믿고 달리면 위험하다 → **상관계수와 같이** 보자.

### 9-2. 개념
| 값 | 의미 | 비유 |
|---|---|---|
| **+1** | 완전한 양의 상관 | 나와 가족 |
| **0** | 상관 없음 | 나와 버스 기사님 (무관심) |
| **-1** | 완전한 음의 상관 | 미움 |

- "사랑의 반대말은 미움이 아니라 **무관심**" → **-1도 관계가 강한 것**이다.
- 직선을 그었을 때 +1이든 -1이든 **둘 다 다음 값을 예측할 수 있다.** 방향만 다를 뿐.
- 그래서 판단 기준은 항상 **절댓값**이다.
- 상관계수(피어슨)는 **선형(linear) 관계**만 본다 → 단순 Linear Regression을 컬럼끼리 해본 것이라고 생각하면 된다. 정확한 지표는 아니다.

### 9-3. 판단 규칙
| 비교 대상 | 절댓값이 높으면 | 절댓값이 낮으면 |
|---|---|---|
| **X ↔ X** (피처끼리) | 사실상 **중복 데이터** → 둘 중 하나 제거를 고민 (1.0이면 거의 무조건) | 괜찮음 |
| **X ↔ Y** (피처와 타겟) | **중요한 피처** | 제거 후보 |

### 9-4. 코드
```python
import matplotlib.pyplot as plt
import seaborn as sns     # pip install seaborn

print(df.corr())

sns.heatmap(data=df.corr(),
            square=True,
            annot=True,     # 칸 안에 숫자 표시
            cbar=True)
plt.show()
```
- `df.corr()` 하나로 끝난다.
- seaborn의 약어가 `sb`가 아니라 **`sns`** 인 건 초창기 개발자들이 그렇게 쓴 게 관행으로 굳어서다. (`plt`, `np`, `pd`, `tf`처럼)

### 9-5. iris 결과 해석
| 관계 | 상관계수 | 해석 |
|---|---|---|
| sepal length ↔ sepal width | 약 -0.11 | 관계 약함 |
| sepal length ↔ petal length | 약 0.87 | 양의 상관 |
| petal length ↔ petal width | 매우 높음 | 거의 같은 데이터(중복) 가능성 |
| petal width ↔ Target | 약 0.95 | **가장 중요한 피처** |
| sepal width ↔ Target | 절댓값 가장 작음 | 컬럼을 솎아내야 한다면 **제거 1순위** |

- 두 번째 실습 데이터에서는 X1과 X2가 극단적으로 상관이 높아 **하나를 빼도 될 것 같고**, X3는 0.039라 애매했다 → **하나씩 제거하며 성능 비교**를 해봐야 한다.

### 9-6. Q&A
- **Q. 상관관계가 높아서 컬럼을 지웠더니 성능이 확 떨어지는 경우도 있나?**
  → 엄청 많다. `y = w1x1 + w2x2 + … + w100x100`에서 x7과 x10이 비슷해서 x7을 지웠을 때 나머지 99개로 성능이 좋아질지 나빠질지 사람이 계산할 수는 없다. **해보고 좋으면 쓰고 아니면 되돌린다.**
  → 그래서 **상관계수 + Feature Importance를 같이** 보고 컬럼을 고른다.

> 보충: iris의 Target은 0/1/2 범주형이라, 타겟과의 상관계수는 대략적인 참고용으로만 보자.

---

## 10. 핵심 요약 (시험/면접용)

1. **`**dict`** = 딕셔너리를 키워드 인자로 언패킹, **`*tuple`** = 튜플을 위치 인자로 언패킹
2. XGBoost 핵심 파라미터: `n_estimators`, `learning_rate`, `max_depth` (+ `random_state`)
3. Early Stopping: `early_stopping_rounds`는 생성자나 `set_params`에, 판단 데이터는 `fit(eval_set=[(x, y)])`
4. 모델 저장: **joblib**(가장 많이) > pickle > XGBoost 자체 `save_model`
5. pickle은 `open(path, 'wb')` / `open(path, 'rb')` 필요
6. XGBoost `load_model`은 **빈 모델을 먼저 정의**해야 함
7. joblib/pickle은 DataFrame의 **인덱스·컬럼명까지** 저장 (numpy는 빠짐)
8. 상관계수는 **절댓값**으로 판단. X끼리 높으면 중복, X-Y 높으면 중요
9. 상관계수는 **선형 관계만** 본다 → Feature Importance와 함께 사용

---

## 11. 진로·생활 팁 (수업 중 이야기)

- **취업용 증명사진은 무난한 게 최고.** 서류를 보는 사람은 대체로 보수적이다. 지원서가 수백 건씩 오면 제목·이력 다음에 사진을 대충 보는데, 개성이 너무 강하면 걸러질 수 있다.
- 기회는 공채로만 오지 않는다. 중소기업에서 시작해 좋은 자리로 옮기는 경우도 있다. **늘 준비하고 있기.**
- 강사님 경험: AI를 쓰면서 작업 속도가 예전보다 4~5배 빨라졌지만, **코드를 이해하면서 쓰기 때문**에 가능한 것.

---

## 12. 다음 일정

- **내일:** 팀 발표 (먼저 자료를 보낸 팀부터 발표 순서 선택권)
- 다음 진도: **PCA(차원 축소)** — 컬럼을 줄여가며 성능 비교
- **다음 주 월요일:** **PyTorch** 시작 (환경 세팅부터, 정각 시작). WSL(리눅스 기반)에서 진행 예정 — 회사가 리눅스 기반인 경우가 많기 때문
- 화요일: 시작 발표 / 금요일: 마지막 공모 발표 → 이후 **매주 화요일 팀 프로젝트 진행 상황 발표**
- 목요일 자정 기준으로 서버 자료가 삭제되니 **자료 백업** 필수. 팀별로 개인 폴더를 만들어 따로 작업
