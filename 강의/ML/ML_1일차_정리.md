# ML 1일차 정리

> 녹취(STT) 기반 정리입니다. 인식 오류가 많아 문맥상 확실한 내용만 추렸고, 일부 보완 설명(💡)을 덧붙였습니다.

## 오늘의 흐름

1. 오전: LangGraph **병렬 처리** 복습 (Search + Analyst → Merge)
2. 오후: 머신러닝 환경 세팅 → 트리 계열 모델 비교 → **Feature Importance**
3. 예고: 내일은 Matplotlib 시각화, 상관관계(Correlation), Hermes 설치

---

## Part 1. LangGraph 병렬 처리 복습

### 1-1. 핵심 개념 3줄 요약

| 개념 | 설명 |
|---|---|
| **State** | 그냥 **딕셔너리(TypedDict)** 라고 생각하면 됨 |
| **Node** | State를 받아서 State에 반영할 값을 돌려주는 **함수** |
| **Edge** | 노드 간 연결. `add_node` → `add_edge` 순서로 붙이면 끝 |

### 1-2. State 정의와 Reducer

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 리스트에 append (reducer)
    search_result: str                        # 웹 검색 결과
    analyst_result: str                       # LLM 분석 결과
```

- `add_messages` = **리듀서(Reducer)**. 개념적으로는 리스트에 **append** 하는 것.
- `Annotated`를 빼고 그냥 `list`로 두면 **노드가 값을 낼 때마다 덮어쓰기** 됨.
- 변수명(`state`, `AAA` 등)은 마음대로 써도 되고, 타입 힌트(`State`)는 가독성용. 실제 동작은 "딕셔너리를 받는다"는 점이 중요.

### 1-3. 노드에서 값 꺼내기

```python
question = state["messages"][-1].content
```

- `state["messages"]` → 메시지 리스트
- `[-1]` → 가장 **마지막** 메시지 (Human/AI 메시지가 계속 쌓이므로 최신 질문은 항상 마지막)
- `.content` → 메타데이터 말고 **본문만** 가져옴

### 1-4. 그래프 구성 (병렬)

- 노드 3개: **search_agent**(웹 검색), **analyst_agent**(LLM 분석), **merge_agent**(결과 통합)
- 기존 방식(노드에서 노드로 순차)과 달리, 시작점에서 **두 노드를 동시에 실행**하고 merge에서 합침
- 구성은 `add_node`, `add_edge`만 반복하면 됨 (Sentence-Transformer 모델 구조처럼 "쭉 내려갔다 올라갔다" 하는 합성 구조와 유사)
- 아직 **메모리(체크포인터)는 안 넣은 상태**. 나중에 금방 추가 가능

```python
graph.add_node("search", search_agent)
graph.add_node("analyst", analyst_agent)
graph.add_node("merge", merge_agent)

graph.add_edge(START, "search")
graph.add_edge(START, "analyst")   # 병렬 실행
graph.add_edge("search", "merge")
graph.add_edge("analyst", "merge")
graph.add_edge("merge", END)
```

### 1-5. 버전 2 → 버전 3 : Merge 에이전트 개선

- **버전 2**: 두 결과를 단순히 **이어 붙이기(concat)** 만 함
- **버전 3**: **프롬프트(f-string)** 로 두 결과를 종합해 최종 답변 생성. 바뀐 건 merge 부분뿐

```python
prompt = f"""
웹 검색 결과: {state["search_result"]}
LLM 분석 결과: {state["analyst_result"]}

위 두 결과를 종합해서 최종 답변을 작성해줘.
검색 결과(최신 정보)를 우선적으로 참고하고, 분석을 덧붙여 설명해줘.
"""
```

**프롬프트 관련 포인트**
- 요즘 모델은 자연어로 말해도 충분히 잘 알아들음 → 복잡한 프롬프트 엔지니어링 기법에 집착할 필요 없음
- 다만 "프롬프트를 바꿨으니 무조건 좋아졌다"고 단정은 어려움. 정량 지표가 없으면 결국 **사람이 보고 판단** (Human Eval)
- 응답 시간이 길어지므로 **시간 측정 코드**를 넣어보면 좋음

### 1-6. 트러블슈팅 (실습 중 나온 문제들)

| 증상 | 원인 / 해결 |
|---|---|
| 통합 에이전트 결과가 비어 있음 | `max_tokens`가 너무 작아 출력이 중간에 끊김 → **3000~5000으로 올리거나**, 아예 설정을 끄기 |
| 모델 바꿔도 안 나옴 | 모델 문제가 아니라 코드 문제일 수 있음 → 브레이크포인트로 각 노드의 입출력 확인 |
| 에러 / 값이 안 들어옴 | **`messages`의 `s` 누락** (`message`로 쓴 오타). State 키 이름과 정확히 일치해야 함 |
| import 에러 | 필요한 모듈 import 누락 확인 |

### 1-7. 기타 (참고)

- **API vs 로컬 모델**: OpenAI API를 쓰면 GPU(NPU) 용량(48GB든 8GB든)은 의미 없음. 로컬 모델(Gemma 32B, Qwen 32B급)을 직접 올려 쓸 때만 GPU 메모리가 의미 있음
- 업무용 에이전트(Hermes, OpenClaw 등)는 자연어 지시만으로 설정 변경·업데이트·롤백까지 수행. 단, 가끔 중간에 죽을 수 있으니 주의
- 에이전트를 여러 개 돌릴 땐 미니 PC 여러 대보다 고성능 데스크탑 1대 + WSL 여러 개가 효율적일 수 있음 (실행 시간대가 달라 전력 효율이 좋음)

---

## Part 2. 머신러닝 환경 세팅

- `study` 폴더 아래 **ML 폴더** 생성, 가상환경 생성 후 활성화 (Python 3.12 환경)
- 설치한 패키지 3개: **XGBoost, scikit-learn, pandas**
- 앞으로 머신러닝 + 데이터 분석은 이 환경에서 진행
- 파일명 규칙 예: `M01_FeatureImportance_01.ipynb`

---

## Part 3. 트리 계열 모델 4종 비교 (Iris)

### 3-1. 모델 계보

| 모델 | 특징 |
|---|---|
| **Decision Tree** | 조건 분기(예: 꽃잎 길이 > x)를 **트리 구조**로 내려가며 분류 |
| **Random Forest** | 트리 여러 개를 모은 **앙상블(숲)** → 단일 트리보다 보통 강력 |
| **Gradient Boosting** | 이전 트리의 오차를 다음 트리가 보완하는 **부스팅** 계열 |
| **XGBoost** | 부스팅 계열의 대표 주자. 정형 데이터에서는 지금도 현역 (딥러닝보다 강한 경우도 많음) |

> 💡 강사님 요지: 예전 레거시 ML 모델은 대부분 밀려났지만, **XGBoost 계열은 아직 현역**. 실무/대회에서 쓰게 될 가능성이 높음.

### 3-2. 데이터 준비

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)   # X, y 한 번에 분리
print(X.shape, y.shape)             # (150, 4), (150,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)
```

- 💡 Iris는 **행(샘플) 150개, 열(feature) 4개**
- `random_state`를 고정하는 이유: 여러 모델을 **같은 조건**에서 비교하기 위해
- `stratify=y`: **분류**에서 클래스 비율을 train/test에 동일하게 유지. y가 연속값인 **회귀에서는 사용 불가 → 제거**

### 3-3. 모델 구성 → 훈련 → 평가 (딥러닝과 비교)

| 단계 | 딥러닝(Keras) | 머신러닝(sklearn) |
|---|---|---|
| 모델 구성 | 레이어 쌓기 | 모델 클래스 생성 (`DecisionTreeClassifier(random_state=123)`) |
| 컴파일 | `compile()` | **없음** |
| 훈련 | `fit()` (epochs 필요) | `fit(X_train, y_train)` (epoch 개념 없음) |
| 평가 | `evaluate()` | **`score(X_test, y_test)`** |

- 파라미터를 몰라도 **default만으로도 충분히 성능이 나옴**
- 파라미터 튜닝은 XGBoost에서 주요 파라미터만 따로 정리해줄 예정

### 3-4. 4개 모델 한 번에 돌리기 (for문)

```python
models = [model1, model2, model3, model4]   # DT, RF, GB, XGB

for model in models:
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(model.__class__.__name__, acc)    # 모델 이름 출력
```

- `model.__class__.__name__` → 모델 이름 출력
- 대략적인 결과: Decision Tree ≈ **0.83**, 나머지 앙상블/부스팅 계열은 **0.9대** (이 데이터에서는 DT가 가장 낮음)
- ※ `random_state` 값을 바꿔가며 "제일 잘 나온 값"만 고르는 건 논문에서 **편향(cherry picking)** 이므로 주의

### 3-5. `score()`의 의미 (중요)

| 모델 종류 | `model.score()` 기준 |
|---|---|
| **Classifier** | **Accuracy** |
| **Regressor** | **R²** |

- 회귀에서 `score`가 1.0에 가깝다면 R² 기준 → 과적합 의심 필요
- 분류 문제를 평가할 때 출력 이름도 `acc`처럼 의미 있게 붙이기

### 3-6. 회귀(Regression)로 바꾸기 (California Housing)

바꿔야 할 것:
- `DecisionTreeClassifier` → `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`
- `train_test_split`의 **`stratify` 제거** (라벨 클래스 수에 따라 조절하는 옵션이라 회귀에선 불가)
- 데이터셋 변경 (`fetch_california_housing` 등)
- 평가 지표가 **R²** 로 바뀜 (R² 점수가 높지 않아도 정상. 어떤 컬럼이 중요한지를 보는 게 목적)

---

## Part 4. Feature Importance (오늘의 핵심)

> 강사님: "프로젝트든 아니든 **피처 엔지니어링은 반드시 할 줄 알아야 한다.**"

### 4-1. 트리 계열 모델의 `feature_importances_`

```python
for model in models:
    model.fit(X_train, y_train)
    print(model.__class__.__name__, model.score(X_test, y_test))
    print(model.feature_importances_)
```

- 트리 계열 모델은 **기본적으로 피처 중요도를 제공**
- 출력 배열의 각 값 = **각 컬럼(feature)의 중요도** (합이 1)
- Iris 결과: 4개 모델 모두 **세 번째 컬럼이 가장 중요**, 첫 번째 컬럼은 거의 0 (예: 0.01 수준)
- 중요도가 낮은 컬럼은 "있으나 마나"일 가능성이 큼

### 4-2. 중요도를 어떻게 활용하나 (Feature Engineering)

- 컬럼이 100~300개면, 소수(약 10개)의 컬럼이 성능을 좌우하는 경우가 흔함
- 중요도 낮은 컬럼을 **Drop**하면 오히려 성능이 좋아질 수 있음 (노이즈 제거)
- 컬럼 수가 매우 많으면(대회 사례: 25,000개) **필요 없는 컬럼 + 성능을 깎는 컬럼**이 많음
- 방법: **삭제(Drop)**, **병합**, **PCA로 합치기**, **파생 피처 생성**

### 4-3. ⚠️ 주의할 점 (질의응답 내용)

- 중요도는 **다른 피처와의 관계까지 고려**해서 계산됨 (단일 피처만 따로 보는 게 아님)
- 컬럼끼리 **상관관계**가 있으면, 하나를 지웠다고 성능이 오른다고 보장할 수 없음
- 컬럼이 4개처럼 적을 땐 하나 지우는 게 **위험**, 수백 개일 땐 수십 개를 지워도 문제없는 경우가 많음
- 보조 수단: **상관관계(Correlation) 분석** (월요일 오후에 별도 진행 예정)
- 최종 판단: **지우고 모델을 다시 돌려서 성능이 좋아졌는지 사람이 확인**

```
스토리 흐름: 피처 중요도 확인 → 낮은 컬럼 Drop → 데이터 재구성 → 모델 재학습 → 성능 향상 확인
```

### 4-4. XGBoost의 중요도 종류 : F score vs Gain

| 종류 | 의미 | 해석 |
|---|---|---|
| **F score** (`importance_type='weight'`) | 트리 분기에서 해당 컬럼이 **사용된 횟수/빈도** | 많이 쓰였다 = 접근이 많았다. 그러나 **성능에 기여했다는 뜻은 아님** |
| **Gain** (`importance_type='gain'`) | 해당 컬럼으로 분기했을 때 **손실함수가 개선된 정도** | 높을수록 **실제 성능 향상에 기여** |

- 💡 이 F score는 분류 평가지표인 **F1-score와는 다른 개념** (F1-score는 나중에 반드시 알아야 할 별개 지표)
- 보통 두 값은 **같이 움직이지만**, 컬럼이 많아지면 순서가 달라질 수 있음
- 둘을 모두 확인한 뒤, 예를 들어 컬럼 100개라면 둘 다 낮은 10~20개를 Drop하는 식으로 활용

### 4-5. XGBoost 중요도 시각화

```python
import matplotlib.pyplot as plt
from xgboost import plot_importance

# F score (기본: weight)
plot_importance(model, importance_type="weight")
plt.show()

# Gain
plot_importance(model, importance_type="gain")
plt.show()
```

- `plot_importance`는 막대 그래프로 바로 그려줌
- 그래프의 `f0, f1, f2, f3` = 첫 번째~네 번째 feature
- 수치 크기는 다를 수 있으나 **상대적 비율/순위**가 중요
- Iris에서는 F score, Gain 모두 **세 번째 feature가 가장 높게** 나옴
- 해커톤/아이디어톤에서 피처 중요도 그래프를 발표 자료로 활용 가능
- 내일은 **Matplotlib으로 직접 그래프 커스터마이징** 예정

---

## 복습 체크리스트

- [ ] LangGraph State = 딕셔너리, `Annotated[list, add_messages]` = append 리듀서
- [ ] 노드 = State를 받아 값을 반환하는 함수, `state["messages"][-1].content`
- [ ] 병렬 구성: `START → (search, analyst) → merge → END`
- [ ] `max_tokens`가 작으면 답변이 끊김, State 키 오타(`messages`) 주의
- [ ] ML 흐름: 모델 생성 → `fit` → `score` (compile, epoch 없음)
- [ ] 분류 `score` = Accuracy, 회귀 `score` = R², 회귀는 `stratify` 제거
- [ ] 트리 모델은 `feature_importances_` 제공
- [ ] XGBoost: **F score = 사용 빈도**, **Gain = 손실 개선도(실질 성능 기여)**
- [ ] 피처를 지운 뒤에는 반드시 **재학습해서 성능 비교**

## 다음 시간 예고

- Matplotlib으로 Feature Importance 그래프 직접 그리기
- Correlation(상관관계) 기반 피처 엔지니어링
- Hermes 설치 (약 30분 예정)
- 오후: 두 팀 → 세 팀으로 늘어난 프로젝트 팀 대상 피처 중요도 적용
