# ML 7일차 정리 — 앙상블 기법 (Bagging · Voting · Stacking)

> 녹취 기반 정리. 오늘 핵심은 **앙상블 4대 기법** 중 배깅, 보팅, 스태킹을 직접 구현하고 비교하는 것.
> (부스팅 계열 = XGBoost / LightGBM / CatBoost는 이전에 배움)

---

## 0. 오늘의 흐름 한눈에 보기

| 기법 | 모델 수 | 데이터 | 결합 방식 | sklearn 클래스 |
|---|---|---|---|---|
| **Bagging** | 1종류 모델 × N번 | 부트스트랩 샘플링 (데이터를 바꿔가며) | 평균 / 다수결 | `BaggingRegressor`, `BaggingClassifier` |
| **Voting** | 여러 종류 모델 | 같은 데이터 | 투표(hard) / 확률 평균(soft) | `VotingClassifier`, `VotingRegressor` |
| **Stacking** | 여러 종류 모델 | 1단계 예측값 → 새 데이터셋 | 2단계 모델(메타 모델)로 재학습 | `StackingClassifier`, `StackingRegressor` |
| **Boosting** | 약한 모델을 순차적으로 | 이전 오차를 보완 | 가중합 | XGBoost, LightGBM, CatBoost |

---

## 1. Bagging (배깅)

### 1-1. 개념
- **모델은 1개 종류만** 사용 (예: `DecisionTreeRegressor`)
- 그 모델을 **데이터를 바꿔가며 여러 번 학습** → 결과를 평균(회귀) 또는 다수결(분류)
- 데이터에서 일정 비율(예: 70%, 80%)을 **랜덤 추출**해 각 모델 학습에 사용

### 1-2. K-Fold와의 차이
| | K-Fold | Bagging (Bootstrap) |
|---|---|---|
| 분할 방식 | 데이터를 K등분, 한 조각씩 검증용으로 뺌 | 전체에서 랜덤 샘플 추출 |
| 중복 | 없음 | **중복 허용 가능** (`bootstrap=True`) |
| 조합 개수 | K개로 고정 | 원하는 만큼 (`n_estimators`) |

예시 (데이터 1~10, 70% 추출):
- 1회차: `1, 2, 3, 4, 4, 6, 7` ← 4가 중복
- 2회차: `2, 3, 3, 5, 6, 9, 10` ← 3이 중복
- 각 회차의 R2 점수(예: 0.7, 0.3, 1.0 …)를 **평균** → 최종 결과

> 중복이 섞이면 과적합될 것 같지만, 트리(Tree) 계열에서는 오히려 은근히 잘 맞는다.

### 1-3. 주요 파라미터
| 파라미터 | 의미 |
|---|---|
| `estimator` | 사용할 기본 모델 (예: `DecisionTreeRegressor()`) |
| `n_estimators` | 모델을 몇 번 돌릴 것인가 (기본 10) |
| `max_samples` | 각 모델에 쓸 샘플 비율 (예: `0.8` → 80%) |
| `bootstrap` | `True` = 중복 허용(기본), `False` = 중복 불허 |
| `n_jobs` | `-1` = CPU 전부 사용 |
| `random_state` | 재현성 확보 |

### 1-4. 코드 (캘리포니아 회귀)
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model = DecisionTreeRegressor()                      # ① 기본 DT
model = BaggingRegressor(DecisionTreeRegressor(),      # ② 배깅
                         n_estimators=100,
                         n_jobs=-1,
                         random_state=42,
                         max_samples=0.8,
                         bootstrap=True)               # False도 비교
# model = RandomForestRegressor(random_state=42)       # ③ 랜덤포레스트

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('R2:', r2_score(y_test, y_pred))
```

### 1-5. 수업 결과
| 모델 | R2 (캘리포니아) |
|---|---|
| DecisionTree (디폴트) | 약 0.5x ~ 0.6 |
| Bagging (`n_estimators=10`) | 약 0.17 이상 상승 |
| Bagging (`n_estimators=100`) | **약 0.807** |
| RandomForest (디폴트) | **약 0.81** |

- `bootstrap=True`가 통상 조금 더 낫다고 하지만, 이번엔 `False`가 더 좋게 나오기도 함 → **실험해보고 좋은 걸 선택**
- `random_state` 값에 따라 점수가 달라짐 → "랜덤의 중요성"

### 1-6. RandomForest = Bagging + α
- **랜덤포레스트는 이미 배깅 기법을 쓰고 있음** (배깅 + 피처까지 랜덤 선택)
- 어설프게 3대장(XGB/LGBM/CAT)을 하이퍼파라미터 튜닝 없이 쓰는 것보다 **RF 디폴트가 더 좋을 때가 많다**

### 1-7. 실습
- `m15_bagging_02_diabetes.py` (회귀) → 디폴트 0.38 정도. `random_state`, `n_estimators`, `max_samples` 조정 시 **0.5 언저리면 괜찮은 수준**
- `m15_bagging_03_cancer.py` (분류) → 바꿀 것:
  - `Regressor` → `Classifier`
  - `train_test_split(..., stratify=y)`
  - `r2_score` → `accuracy_score`

> 배깅 = **한 가지 모델**로 데이터만 요리조리 바꿔가며 여러 번 돌리는 것

---

## 2. Voting (보팅)

### 2-1. 개념
- **서로 다른 여러 모델**을 같은 데이터로 학습 → 결과를 **투표**로 결정
- 예: XGBoost, LightGBM, CatBoost 3개가 이진분류
  - 0, 0, 1 → **0**
  - 1, 1, 0 → **1**

> 다수결의 위험성 비유: "늑대 2마리 + 양 1마리가 저녁 메뉴를 투표로 정하면?" → 소수에게 위험할 수 있음

### 2-2. Hard Voting vs Soft Voting
**Hard Voting** — 각 모델의 최종 라벨(0/1)로 다수결
- 모델 개수가 **짝수(4개)** 면 2:2 동률 발생
- 동률 시 `argmax`는 **앞에 있는 값**을 선택 → 성능 저하 가능
- 해결: 홀수 개 모델 사용 or Soft Voting

**Soft Voting** — 각 모델의 **클래스별 확률을 평균**해서 큰 쪽 선택
```
        클래스0  클래스1
XGB      0.7     0.3
LGBM     0.4     0.6
CAT      0.8     0.2
RF       0.5     0.5
---------------------
평균     0.6     0.4   → 결과: 0
```
- 통상 **Soft Voting이 더 많이 쓰이고 성능도 더 좋음**
- `VotingClassifier`의 기본값은 `voting='hard'` (설명이 쉬워서인 듯)

### 2-3. 라이브러리 설치
```bash
pip install lightgbm
pip install catboost
```

### 2-4. 코드 (유방암 분류)
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=3444, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xgb = XGBClassifier()
lg  = LGBMClassifier(verbose=-1)   # 로그 끄기
cat = CatBoostClassifier(verbose=0) # 로그 끄기
rf  = RandomForestClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat), ('RF', rf)],  # (이름, 모델) 튜플 리스트
    voting='soft',   # 'hard'(기본) / 'soft'
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('ACC:', accuracy_score(y_test, y_pred))
```

- `estimators`는 **(이름, 모델) 튜플의 리스트** → 이름을 붙여야 나중에 파라미터 지정 가능
- 수업 결과: hard 0.96 → `random_state=3444`에서 hard/soft 모두 **0.9912** (데이터가 작아서 차이 적음)

### 2-5. Q&A 정리
- **Q. 하이퍼파라미터 튜닝 후 보팅하면 더 좋아지나?**
  → 예. 지금은 디폴트라 각 모델 성능이 75~77% 수준일 수 있음. **튜닝된 모델끼리** 4개든 6개든 10개든 묶으면 더 강해짐
- **Q. Regressor는 Hard/Soft 중 뭐가 좋은가?**
  → **회귀에는 Hard Voting이 존재하지 않음.** 출력이 0/1 같은 라벨이 아니라 0.237 같은 연속값이므로 투표 불가 → **평균**만 가능. 그래서 `VotingRegressor`에는 `voting` 파라미터 자체가 없음

```python
from sklearn.ensemble import VotingRegressor
model = VotingRegressor(
    estimators=[('XGB', XGBRegressor()), ('LG', LGBMRegressor(verbose=-1)),
                ('CAT', CatBoostRegressor(verbose=0)), ('RF', RandomForestRegressor())],
)
```

### 2-6. 실습
- `m16_voting_02_california.py`, `m16_voting_03_diabetes.py` (회귀)
- 결과: 캘리포니아는 좋아짐, 디아벳은 나빠짐 → **무조건 좋아지는 기법은 없다**

### 2-7. 현업 이야기
- "무조건 딥러닝이 최고"는 아님
- **은행·금융·증권 등 정형 데이터 분야**에서는 여전히 레거시 ML(3대장 모델)이 훨씬 강력함

### 2-8. verbose 정리
| 모델 | 로그 끄는 법 |
|---|---|
| LightGBM | `verbose=-1` |
| CatBoost | `verbose=0` |
| XGBoost | 기본적으로 조용함 (`verbosity=0`) |

---

## 3. Stacking (스태킹)

### 3-1. 개념
- **모델을 두 번 돌린다**
  1. 1단계: 여러 모델로 학습 → **예측값(y_hat)** 을 뽑음
  2. 2단계: 그 예측값들을 **새로운 X(데이터셋)** 로 삼아 원래 y를 다시 학습
- 이미 배운 적 있음 → **결측치 처리 때 모델로 예측해서 채우는 방법** (`IterativeImputer`)도 일종의 스태킹

```
원본 X (n, 100컬럼) ──▶ XGB  ──▶ y_hat1 ┐
                    ──▶ LGBM ──▶ y_hat2 ├─▶ 새 X (n, 4) ──▶ 메타 모델 ──▶ 최종 y
                    ──▶ CAT  ──▶ y_hat3 │
                    ──▶ RF   ──▶ y_hat4 ┘
```

- 100개 컬럼 → 4개 컬럼: 일종의 **차원 축소** 효과
- 원본 피처를 포함할지, 예측값만 쓸지는 자유

> 용어 정리: **열 = 컬럼 = 피처 = 특성 = 속성 = 어트리뷰트 = 차원** (모두 같은 말)
> 차원 축소 기법 중 대표: **PCA** (아직 안 배움, 오래된 기법이라 성능 보장은 어려움)

### 3-2. 보팅/결측치 처리와의 차이
- **결측치 처리**: 빈 공간만 모델로 채움
- **보팅**: 모델 결과를 투표/평균
- **스태킹**: 예측값으로 **아예 새 데이터셋을 만들고** 그걸로 재학습

### 3-3. 스태킹 사용 팁
- 성능이 **확 좋아지지 않음.** 아주 미세하게 올라감 (떨어질 수도 있음)
- 그래도 **컴피티션 후반부 싸움**에서 유용 → 0.01 사이에 수천 명이 몰려 있으므로
- **같은 모델 + 같은 파라미터를 두 번 쓰면 과적합 → 의미 없음**
  - 다른 모델을 쓰거나
  - 같은 모델이면 **하이퍼파라미터를 완전히 다르게** 튜닝
- 1단계/2단계 모두 강한 모델이 가장 좋음. 최종 결과는 2단계에서 나오므로 **2단계는 반드시 강해야 함**

### 3-4. 직접 구현 (생코딩) — `m17_stacking_01_cancer.py`
```python
import numpy as np
from sklearn.metrics import accuracy_score

models = [xgb, lg, cat, rf]   # 3대장 먼저, 중장급(RF)은 뒤로

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)   # train 예측값
    y_test_pred  = model.predict(x_test)    # test 예측값

    train_list.append(y_train_pred)
    test_list.append(y_test_pred)

    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__   # 모델 클래스 이름 문자열
    print('{0} ACC : {1:.4f}'.format(class_name, score))

# 리스트 → numpy, 행/열 뒤집기
x_train_new = np.array(train_list).T   # (4, n) → (n, 4)
x_test_new  = np.array(test_list).T
print(x_train_new.shape, x_test_new.shape)

# 2단계(메타) 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred2)
print('스태킹 결과 :', score2)
```

**포인트**
- `np.array(list)`는 `(모델 수, 샘플 수)` = `(4, n)` 형태 → `.T`로 전치해서 `(n, 4)`로
- `'{0} ... {1:.4f}'.format(a, b)` → 0번째/1번째 자리에 변수 대입, `.4f`는 소수 넷째 자리
- `model.__class__.__name__` → `'XGBClassifier'` 같은 클래스 이름 문자열
- 수업 결과: 개별 모델 0.95~0.99, 스태킹 **0.9912** (CatBoost 단독과 동일) → 이미 1단계에 같은 CatBoost가 들어 있어 사실상 **동일 모델 두 번 사용 → 개선 없음**

> ⚠️ 보충: 위 방식은 1단계에서 **train 데이터로 학습하고 같은 train을 예측**하므로 예측값이 지나치게 정답에 가깝게 나옴(데이터 누수 → 과적합). 정석은 `cross_val_predict`로 **폴드 밖(out-of-fold) 예측값**을 만드는 것이고, 아래 `StackingClassifier`가 `cv` 파라미터로 이걸 자동 처리함.
> ```python
> from sklearn.model_selection import cross_val_predict
> y_train_pred = cross_val_predict(model, x_train, y_train, cv=5)
> ```

### 3-5. sklearn 버전 — `m18_stacking_01_cancer.py`
```python
from sklearn.ensemble import StackingClassifier

model = StackingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat), ('RF', rf)],
    final_estimator=CatBoostClassifier(verbose=0),  # 메타 모델
    n_jobs=-1,
    cv=5,          # 정수만 넣어도 KFold(5) 자동 적용
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('ACC:', accuracy_score(y_test, y_pred))
```
- 파라미터 구조는 Voting/Bagging과 거의 동일: `estimators`는 (이름, 모델) 튜플 리스트
- `final_estimator`: 2단계 모델. 분류 문제면 **Classifier**를 써야 함 (Regressor 넣으면 안 됨)
- `cv`: 따로 `KFold`를 정의하지 않아도 `cv=5`만 쓰면 됨
- 실습: `m18_stacking_02_california.py`, `m18_stacking_03_diabetes.py` → `StackingRegressor` 사용

---

## 4. 성능을 더 올리려면? (튜닝·데이터)

### 4-1. 하이퍼파라미터 자동 튜닝 도구
- **GridSearchCV / HalvingGridSearchCV**: 조건 조합을 만들어 탐색
- **Optuna**, **Bayesian Optimization**: 알고리즘 기반 탐색 → 최근 Optuna 위주로 사용
- 그 외 다양한 AutoML 도구 존재

### 4-2. 진짜 중요한 건 데이터 전처리
- 모델을 아무리 돌려도 **0.01 올리기가 매우 어려움**
- 경쟁자와의 차이를 뒤집는 유일한 방법은 **데이터 쪽 작업**
- **SMOTE (데이터 증폭)** → 다음 시간에 배움
  - 데이터를 증폭한 뒤 다시 학습
  - 강사님 경험: 데이터 분석 대회에서 SMOTE로 상위권 장기 유지

> 앙상블 실력 + 하이퍼파라미터 튜닝 + **데이터 전처리(증폭)** = 상위권

---

## 5. 오늘 만든 파일 정리

| 파일 | 내용 |
|---|---|
| `m15_bagging_01_california.py` | DT vs Bagging(bootstrap T/F) vs RF |
| `m15_bagging_02_diabetes.py` | 배깅 회귀 실습 |
| `m15_bagging_03_cancer.py` | 배깅 분류 실습 |
| `m16_voting_01_cancer.py` | VotingClassifier hard/soft |
| `m16_voting_02_california.py` | VotingRegressor |
| `m16_voting_03_diabetes.py` | VotingRegressor |
| `m17_stacking_01_cancer.py` | 스태킹 직접 구현 |
| `m18_stacking_01_cancer.py` | StackingClassifier |
| `m18_stacking_02/03` | StackingRegressor (캘리포니아·디아벳) |

---

## 6. 핵심 요약 (시험/면접용)

1. **Bagging**: 한 모델 × 여러 부트스트랩 샘플 → 평균/다수결. RandomForest가 대표 (배깅 + 피처 랜덤)
2. **Bootstrap**: 중복을 허용한 랜덤 샘플링. K-Fold와 달리 조합 수 제한 없음
3. **Voting**: 여러 모델 투표. Hard = 라벨 다수결, Soft = 확률 평균. 보통 Soft가 더 좋음
4. **회귀는 Hard Voting 불가** → 평균만 (`VotingRegressor`에 voting 파라미터 없음)
5. **Hard Voting 짝수 모델** → 동률 시 argmax가 앞의 값 선택 → 성능 저하 위험
6. **Stacking**: 1단계 예측값으로 새 데이터셋 → 2단계 메타 모델. 동일 모델·동일 파라미터 반복은 과적합
7. 스태킹은 **미세한 향상** → 컴피티션 후반부용. 본질적 향상은 **데이터 전처리**에서
8. 튜닝 안 한 3대장보다 **RF 디폴트가 더 나을 수 있다**
9. 정형 데이터(금융 등)에서는 여전히 **ML 3대장**이 강력

---

## 7. 다음 일정
- 다음 시간: **SMOTE(데이터 증폭)**, ML 모델 저장(3대장 save)
- 이후: **PyTorch** 시작 — **WSL 설치**부터 진행 (9시 정각 시작)
- 목요일 발표 → POC 완료 후 팀 프로젝트 시작
